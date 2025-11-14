// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: Apache-2.0

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// 	http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! # Remote Externalities
//!
//! An equivalent of `sp_io::TestExternalities` that can load its state from a remote substrate
//! based chain, or a local state snapshot file.

mod logging;

use codec::{Compact, Decode, Encode};
use indicatif::{ProgressBar, ProgressStyle};
use jsonrpsee::{
	core::params::ArrayParams,
	ws_client::{WsClient, WsClientBuilder},
};
use log::*;
use serde::de::DeserializeOwned;
use sp_core::{
	hexdisplay::HexDisplay,
	storage::{
		well_known_keys::{is_default_child_storage_key, DEFAULT_CHILD_STORAGE_KEY_PREFIX},
		ChildInfo, ChildType, PrefixedStorageKey, StorageData, StorageKey,
	},
};
use sp_runtime::{
	traits::{Block as BlockT, HashingFor},
	StateVersion,
};
use sp_state_machine::TestExternalities;
use std::{
	cmp::{max, min},
	collections::VecDeque,
	fs,
	ops::{Deref, DerefMut},
	path::{Path, PathBuf},
	sync::{Arc, Mutex},
	time::{Duration, Instant},
};
use substrate_rpc_client::{rpc_params, BatchRequestBuilder, ChainApi, ClientT, StateApi};
use tokio_retry::{strategy::FixedInterval, Retry};

type Result<T, E = &'static str> = std::result::Result<T, E>;

type KeyValue = (StorageKey, StorageData);
type TopKeyValues = Vec<KeyValue>;
type ChildKeyValues = Vec<(ChildInfo, Vec<KeyValue>)>;
type SnapshotVersion = Compact<u16>;

/// Represents a range of keys to fetch from the remote node.
#[derive(Debug, Clone)]
struct KeyRange {
	/// The starting key of this range (inclusive).
	start_key: StorageKey,
	/// The ending key of this range (exclusive), or None for open-ended range.
	end_key: Option<StorageKey>,
	/// The common prefix for this range.
	prefix: StorageKey,
}

impl KeyRange {
	fn new(start_key: StorageKey, end_key: Option<StorageKey>, prefix: StorageKey) -> Self {
		Self { start_key, end_key, prefix }
	}
}

/// Thread-safe work queue for distributing key ranges to workers.
type WorkQueue = Arc<Mutex<VecDeque<KeyRange>>>;

/// Manages WebSocket client connections for parallel workers.
struct ConnectionManager {
	transports: Vec<Arc<tokio::sync::Mutex<Transport>>>,
}

impl ConnectionManager {
	fn new(transports: Vec<Transport>) -> Result<Self> {
		if transports.is_empty() {
			return Err("At least one transport must be provided");
		}

		Ok(Self {
			transports: transports
				.into_iter()
				.map(|t| Arc::new(tokio::sync::Mutex::new(t)))
				.collect(),
		})
	}

	/// Get a client for a specific worker. Distributes workers across available transports.
	async fn get_client(&self, worker_index: usize) -> Arc<WsClient> {
		let transport_index = worker_index % self.transports.len();
		let transport = self.transports[transport_index].lock().await;
		transport.client.clone()
	}

	async fn recreate_client(&self, worker_index: usize) -> Result<()> {
		let transport_index = worker_index % self.transports.len();

		let mut transport = self.transports[transport_index].lock().await;

		let uri = transport.uri().ok_or("No URI available for client recreation")?;
		warn!(target: LOG_TARGET, "Worker {} recreating WebSocket client connection to {}", worker_index, uri);

		transport.recreate().await?;

		info!(target: LOG_TARGET, "Successfully recreated WebSocket client for worker {}", worker_index);
		Ok(())
	}
}

const LOG_TARGET: &str = "remote-ext";
const DEFAULT_HTTP_ENDPOINT: &str = "https://try-runtime.polkadot.io:443";
const SNAPSHOT_VERSION: SnapshotVersion = Compact(4);

/// The snapshot that we store on disk.
#[derive(Decode, Encode)]
struct Snapshot<B: BlockT> {
	snapshot_version: SnapshotVersion,
	state_version: StateVersion,
	// <Vec<Key, (Value, MemoryDbRefCount)>>
	raw_storage: Vec<(Vec<u8>, (Vec<u8>, i32))>,
	// The storage root of the state. This may vary from the storage root in the header, if not the
	// entire state was fetched.
	storage_root: B::Hash,
	header: B::Header,
}

impl<B: BlockT> Snapshot<B> {
	pub fn new(
		state_version: StateVersion,
		raw_storage: Vec<(Vec<u8>, (Vec<u8>, i32))>,
		storage_root: B::Hash,
		header: B::Header,
	) -> Self {
		Self {
			snapshot_version: SNAPSHOT_VERSION,
			state_version,
			raw_storage,
			storage_root,
			header,
		}
	}

	fn load(path: &PathBuf) -> Result<Snapshot<B>> {
		let bytes = fs::read(path).map_err(|_| "fs::read failed.")?;
		// The first item in the SCALE encoded struct bytes is the snapshot version. We decode and
		// check that first, before proceeding to decode the rest of the snapshot.
		let snapshot_version = SnapshotVersion::decode(&mut &*bytes)
			.map_err(|_| "Failed to decode snapshot version")?;

		if snapshot_version != SNAPSHOT_VERSION {
			return Err("Unsupported snapshot version detected. Please create a new snapshot.")
		}

		Decode::decode(&mut &*bytes).map_err(|_| "Decode failed")
	}
}

/// An externalities that acts exactly the same as [`sp_io::TestExternalities`] but has a few extra
/// bits and pieces to it, and can be loaded remotely.
pub struct RemoteExternalities<B: BlockT> {
	/// The inner externalities.
	pub inner_ext: TestExternalities<HashingFor<B>>,
	/// The block header which we created this externality env.
	pub header: B::Header,
}

impl<B: BlockT> Deref for RemoteExternalities<B> {
	type Target = TestExternalities<HashingFor<B>>;
	fn deref(&self) -> &Self::Target {
		&self.inner_ext
	}
}

impl<B: BlockT> DerefMut for RemoteExternalities<B> {
	fn deref_mut(&mut self) -> &mut Self::Target {
		&mut self.inner_ext
	}
}

/// The execution mode.
#[derive(Clone)]
pub enum Mode<H> {
	/// Online. Potentially writes to a snapshot file.
	Online(OnlineConfig<H>),
	/// Offline. Uses a state snapshot file and needs not any client config.
	Offline(OfflineConfig),
	/// Prefer using a snapshot file if it exists, else use a remote server.
	OfflineOrElseOnline(OfflineConfig, OnlineConfig<H>),
}

impl<H> Default for Mode<H> {
	fn default() -> Self {
		Mode::Online(OnlineConfig::default())
	}
}

/// Configuration of the offline execution.
///
/// A state snapshot config must be present.
#[derive(Clone)]
pub struct OfflineConfig {
	/// The configuration of the state snapshot file to use. It must be present.
	pub state_snapshot: SnapshotConfig,
}

/// Description of the transport protocol (for online execution).
#[derive(Debug, Clone)]
pub struct Transport {
	client: Arc<WsClient>,
	uri: String,
}

impl Transport {
	/// Create a WebSocket client for the given URI.
	///
	/// This is shared between initial creation and reconnection logic.
	pub(crate) async fn create_client(uri: &str) -> Result<WsClient> {
		debug!(target: LOG_TARGET, "initializing remote client to {:?}", uri);

		WsClientBuilder::default()
			.max_request_size(u32::MAX)
			.max_response_size(u32::MAX)
			.request_timeout(std::time::Duration::from_secs(60 * 5))
			.build(uri)
			.await
			.map_err(|e| {
				error!(target: LOG_TARGET, "error: {e:?}");
				"failed to build ws client"
			})
	}

	/// Create a new Transport from a URI, establishing the WebSocket connection.
	pub async fn new(uri: impl Into<String>) -> Result<Self> {
		let uri = uri.into();
		let ws_client = Self::create_client(&uri).await?;
		Ok(Self { client: Arc::new(ws_client), uri })
	}

	/// Recreate the WebSocket client using the stored URI.
	async fn recreate(&mut self) -> Result<()> {
		let ws_client = Self::create_client(&self.uri).await?;
		self.client = Arc::new(ws_client);
		Ok(())
	}

	fn as_client(&self) -> &WsClient {
		&self.client
	}

	fn uri(&self) -> Option<&str> {
		if !self.uri.is_empty() {
			Some(&self.uri)
		} else {
			None
		}
	}
}

impl From<WsClient> for Transport {
	fn from(client: WsClient) -> Self {
		// When constructing from WsClient directly, we don't have the URI
		// This is fine for non-parallel use cases
		Transport { client: Arc::new(client), uri: String::new() }
	}
}

/// Configuration of the online execution.
///
/// A state snapshot config may be present and will be written to in that case.
#[derive(Clone)]
pub struct OnlineConfig<H> {
	/// The block hash at which to get the runtime state. Will be latest finalized head if not
	/// provided.
	pub at: Option<H>,
	/// An optional state snapshot file to WRITE to, not for reading. Not written if set to `None`.
	pub state_snapshot: Option<SnapshotConfig>,
	/// The pallets to scrape. These values are hashed and added to `hashed_prefix`.
	pub pallets: Vec<String>,
	/// Transport URIs. Can be a single URI or multiple for load distribution.
	pub transport_uris: Vec<String>,
	/// Initialized transports (created from transport_uris during initialization).
	transports: Vec<Transport>,
	/// Lookout for child-keys, and scrape them as well if set to true.
	pub child_trie: bool,
	/// Storage entry key prefixes to be injected into the externalities. The *hashed* prefix must
	/// be given.
	pub hashed_prefixes: Vec<Vec<u8>>,
	/// Storage entry keys to be injected into the externalities. The *hashed* key must be given.
	pub hashed_keys: Vec<Vec<u8>>,
}

impl<H: Clone> OnlineConfig<H> {
	/// Return rpc (ws) client reference. Uses the first transport for non-parallel operations.
	fn rpc_client(&self) -> &WsClient {
		self.transports
			.get(0)
			.expect("at least one transport must be configured; qed.")
			.as_client()
	}

	fn at_expected(&self) -> H {
		self.at.clone().expect("block at must be initialized; qed")
	}
}

impl<H> Default for OnlineConfig<H> {
	fn default() -> Self {
		Self {
			transport_uris: vec![DEFAULT_HTTP_ENDPOINT.to_owned()],
			transports: vec![],
			child_trie: true,
			at: None,
			state_snapshot: None,
			pallets: Default::default(),
			hashed_keys: Default::default(),
			hashed_prefixes: Default::default(),
		}
	}
}

impl<H> From<String> for OnlineConfig<H> {
	fn from(uri: String) -> Self {
		Self { transport_uris: vec![uri], ..Default::default() }
	}
}

/// Configuration of the state snapshot.
#[derive(Clone)]
pub struct SnapshotConfig {
	/// The path to the snapshot file.
	pub path: PathBuf,
}

impl SnapshotConfig {
	pub fn new<P: Into<PathBuf>>(path: P) -> Self {
		Self { path: path.into() }
	}
}

impl From<String> for SnapshotConfig {
	fn from(s: String) -> Self {
		Self::new(s)
	}
}

impl Default for SnapshotConfig {
	fn default() -> Self {
		Self { path: Path::new("SNAPSHOT").into() }
	}
}

/// Builder for remote-externalities.
#[derive(Clone)]
pub struct Builder<B: BlockT> {
	/// Custom key-pairs to be injected into the final externalities. The *hashed* keys and values
	/// must be given.
	hashed_key_values: Vec<KeyValue>,
	/// The keys that will be excluded from the final externality. The *hashed* key must be given.
	hashed_blacklist: Vec<Vec<u8>>,
	/// Connectivity mode, online or offline.
	mode: Mode<B::Hash>,
	/// If provided, overwrite the state version with this. Otherwise, the state_version of the
	/// remote node is used. All cache files also store their state version.
	///
	/// Overwrite only with care.
	overwrite_state_version: Option<StateVersion>,
}

impl<B: BlockT> Default for Builder<B> {
	fn default() -> Self {
		Self {
			mode: Default::default(),
			hashed_key_values: Default::default(),
			hashed_blacklist: Default::default(),
			overwrite_state_version: None,
		}
	}
}

// Mode methods
impl<B: BlockT> Builder<B> {
	fn as_online(&self) -> &OnlineConfig<B::Hash> {
		match &self.mode {
			Mode::Online(config) => config,
			Mode::OfflineOrElseOnline(_, config) => config,
			_ => panic!("Unexpected mode: Online"),
		}
	}

	fn as_online_mut(&mut self) -> &mut OnlineConfig<B::Hash> {
		match &mut self.mode {
			Mode::Online(config) => config,
			Mode::OfflineOrElseOnline(_, config) => config,
			_ => panic!("Unexpected mode: Online"),
		}
	}
}

// RPC methods
impl<B: BlockT> Builder<B>
where
	B::Hash: DeserializeOwned,
	B::Header: DeserializeOwned,
{
	const PARALLEL_REQUESTS: usize = 24;
	const BATCH_SIZE_INCREASE_FACTOR: f32 = 1.10;
	const BATCH_SIZE_DECREASE_FACTOR: f32 = 0.50;
	const REQUEST_DURATION_TARGET: Duration = Duration::from_secs(15);
	const INITIAL_BATCH_SIZE: usize = 10;
	// nodes by default will not return more than 1000 keys per request
	const DEFAULT_KEY_DOWNLOAD_PAGE: u32 = 1000;
	const MAX_RETRIES: usize = 12;
	const KEYS_PAGE_RETRY_INTERVAL: Duration = Duration::from_secs(5);

	async fn rpc_get_storage(
		&self,
		key: StorageKey,
		maybe_at: Option<B::Hash>,
	) -> Result<Option<StorageData>> {
		trace!(target: LOG_TARGET, "rpc: get_storage");
		self.as_online().rpc_client().storage(key, maybe_at).await.map_err(|e| {
			error!(target: LOG_TARGET, "Error = {e:?}");
			"rpc get_storage failed."
		})
	}

	/// Get the latest finalized head.
	async fn rpc_get_head(&self) -> Result<B::Hash> {
		trace!(target: LOG_TARGET, "rpc: finalized_head");

		// sadly this pretty much unreadable...
		ChainApi::<(), _, B::Header, ()>::finalized_head(self.as_online().rpc_client())
			.await
			.map_err(|e| {
				error!(target: LOG_TARGET, "Error = {e:?}");
				"rpc finalized_head failed."
			})
	}

	/// Get a single page of keys using a specific client.
	async fn get_keys_single_page_with_client(
		&self,
		client: &WsClient,
		prefix: Option<StorageKey>,
		start_key: Option<StorageKey>,
		at: B::Hash,
	) -> Result<Vec<StorageKey>> {
		client
			.storage_keys_paged(prefix, Self::DEFAULT_KEY_DOWNLOAD_PAGE, start_key, Some(at))
			.await
			.map_err(|e| {
				error!(target: LOG_TARGET, "Error = {e:?}");
				"rpc get_keys failed"
			})
	}

	/// Generate start keys for parallel fetching, dividing the workload.
	/// Uses the same logic as the original gen_start_keys but returns KeyRange objects.
	fn gen_key_ranges(prefix: &StorageKey) -> Vec<KeyRange> {
		let mut prefix_bytes = prefix.as_ref().to_vec();
		let scale = 32usize.saturating_sub(prefix_bytes.len());

		// No need to divide workload if key space is small
		if scale < 9 {
			prefix_bytes.resize(32, 0);
			return vec![KeyRange::new(StorageKey(prefix_bytes.clone()), None, prefix.clone())];
		}

		let chunks = 16;
		let step = 0x10000 / chunks;
		let ext = scale - 2;

		let mut ranges = Vec::with_capacity(chunks);
		for i in 0..chunks {
			let mut start_key = prefix_bytes.clone();
			let start = i * step;
			start_key.extend(vec![(start >> 8) as u8, (start & 0xff) as u8]);
			start_key.extend(vec![0; ext]);

			let end_key = if i < chunks - 1 {
				let mut end_key = prefix_bytes.clone();
				let end = (i + 1) * step;
				end_key.extend(vec![(end >> 8) as u8, (end & 0xff) as u8]);
				end_key.extend(vec![0; ext]);
				Some(StorageKey(end_key))
			} else {
				None
			};

			ranges.push(KeyRange::new(StorageKey(start_key), end_key, prefix.clone()));
		}

		ranges
	}

	/// Initialize the work queue with ranges for each prefix.
	fn initialize_work_queue(prefixes: &[StorageKey]) -> WorkQueue {
		let mut queue = VecDeque::new();

		for prefix in prefixes {
			let ranges = Self::gen_key_ranges(prefix);
			queue.extend(ranges);
		}

		Arc::new(Mutex::new(queue))
	}

	/// Get keys with `prefix` at `block` in a parallel manner using dynamic work queue.
	///
	/// This implementation uses a truly dynamic work queue approach:
	/// 1. Start with 16 initial ranges (dividing the key space)
	/// 2. Workers fetch ONE batch (1000 keys) from a range
	/// 3. If the batch is full, subdivide the REMAINING key space into 16 new ranges
	/// 4. Add the new ranges back to the queue (queue grows dynamically)
	/// 5. Workers continue pulling from the queue until it's empty and all batches are incomplete
	///
	/// This adapts to key density: dense regions get subdivided more, sparse regions less.
	async fn rpc_get_keys_parallel(
		&self,
		prefix: &StorageKey,
		block: B::Hash,
		parallel: usize,
	) -> Result<Vec<StorageKey>> {
		// Initialize work queue with top-level 16 ranges for this prefix
		let work_queue = Self::initialize_work_queue(&[prefix.clone()]);
		let initial_ranges = work_queue.lock().unwrap().len();
		eprintln!("🔧 Initialized work queue with {} ranges for parallel fetching", initial_ranges);

		// Create connection manager for handling client recreation across multiple RPC providers
		let online_config = self.as_online();
		let conn_manager = Arc::new(ConnectionManager::new(online_config.transports.clone())?);
		eprintln!(
			"🌐 Using {} RPC provider(s) for parallel fetching",
			online_config.transports.len()
		);

		// Shared storage for all collected keys
		let all_keys: Arc<Mutex<Vec<StorageKey>>> = Arc::new(Mutex::new(Vec::new()));

		// Track progress logging (log every 10,000 keys)
		let last_logged_milestone = Arc::new(std::sync::atomic::AtomicUsize::new(0));

		// Semaphore to limit parallel workers
		let semaphore = Arc::new(tokio::sync::Semaphore::new(parallel));
		let builder = Arc::new(self.clone());

		// Track active workers
		let active_workers = Arc::new(std::sync::atomic::AtomicUsize::new(0));

		let mut handles = vec![];

		// Spawn worker tasks
		eprintln!("🚀 Spawning {} parallel workers for key fetching", parallel);

		for worker_index in 0..parallel {
			let permit =
				semaphore.clone().acquire_owned().await.expect("semaphore should not be closed");

			let builder = builder.clone();
			let work_queue = work_queue.clone();
			let all_keys = all_keys.clone();
			let active_workers = active_workers.clone();
			let conn_manager = conn_manager.clone();
			let last_logged_milestone = last_logged_milestone.clone();

			let handle = tokio::spawn(async move {
				eprintln!("👷 Worker {} started", worker_index);
				let mut is_active = false; // Track whether this worker is counted as active

				loop {
					// Try to get work from the queue
					let maybe_range = {
						let mut queue = work_queue.lock().unwrap();
						queue.pop_front()
					};

					let range = match maybe_range {
						Some(r) => {
							// Got work - if we weren't active, become active now
							if !is_active {
								active_workers.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
								is_active = true;
							}
							r
						},
						None => {
							// No work available - if we were active, become idle now
							if is_active {
								active_workers.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
								is_active = false;
							}

							// Small delay to allow other workers to potentially add more work
							tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

							// Check again if there's new work or if all workers are idle
							let queue_len = work_queue.lock().unwrap().len();
							let active = active_workers.load(std::sync::atomic::Ordering::SeqCst);

							if queue_len == 0 && active == 0 {
								// No work and no active workers - we're done
								break;
							} else {
								// Either queue has work or other workers are still active - keep
								// waiting
								continue;
							}
						},
					};

					// Get the client for this worker (distributed across RPC providers)
					let client = conn_manager.get_client(worker_index).await;

					// Process this range - fetch ONE batch
					match builder
						.rpc_get_keys_single_batch(range.clone(), block, worker_index, &client)
						.await
					{
						Ok((batch_keys, is_full_batch)) => {
							let last_key = batch_keys.last().cloned();

							// Store the keys we found
							let total_keys = {
								let mut keys = all_keys.lock().unwrap();
								keys.extend(batch_keys);
								keys.len()
							};

							// Log progress every 10,000 keys
							const LOG_INTERVAL: usize = 10_000;
							let current_milestone = (total_keys / LOG_INTERVAL) * LOG_INTERVAL;
							let last_milestone =
								last_logged_milestone.load(std::sync::atomic::Ordering::Relaxed);

							if current_milestone > last_milestone && current_milestone > 0 {
								if last_logged_milestone
									.compare_exchange(
										last_milestone,
										current_milestone,
										std::sync::atomic::Ordering::SeqCst,
										std::sync::atomic::Ordering::Relaxed,
									)
									.is_ok()
								{
									eprintln!("📊 Scraped {} keys so far...", total_keys);
								}
							}

							// If we got a full batch, subdivide the remaining key space
							if is_full_batch {
								if let Some(last) = last_key {
									let new_ranges = Self::subdivide_remaining_range(
										&last,
										range.end_key.as_ref(),
										&range.prefix,
									);

									if !new_ranges.is_empty() {
										debug!(
											target: LOG_TARGET,
											"Worker {worker_index}: subdividing remaining range after {:?} into {} new ranges",
											HexDisplay::from(&last),
											new_ranges.len()
										);

										let mut queue = work_queue.lock().unwrap();
										queue.extend(new_ranges);
									}
								}
							}

							// Small delay to avoid overwhelming the node
							tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
						},
						Err(e) => {
							warn!(
								target: LOG_TARGET,
								"Worker {worker_index} failed to fetch keys: {e:?}. Attempting to recreate client..."
							);

							// Try to recreate the WebSocket client for this worker's provider
							if let Err(recreate_err) =
								conn_manager.recreate_client(worker_index).await
							{
								error!(
									target: LOG_TARGET,
									"Worker {worker_index} failed to recreate client: {recreate_err:?}"
								);
							}

							// Put the range back in the queue for retry
							{
								let mut queue = work_queue.lock().unwrap();
								queue.push_back(range);
							}

							// Wait to avoid hammering a potentially failing connection
							tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
						},
					}
				}

				// Cleanup: if we're still marked as active, decrement before exiting
				if is_active {
					active_workers.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
				}

				eprintln!("✅ Worker {} finished", worker_index);
				drop(permit);
			});

			handles.push(handle);
		}

		// Wait for all workers to complete
		futures::future::join_all(handles).await;

		// Extract and return all keys
		let keys = all_keys.lock().unwrap().clone();
		eprintln!(
			"🎉 Parallel key fetching complete: {} total keys fetched by {} workers",
			keys.len(),
			parallel
		);

		Ok(keys)
	}

	/// Get ONE batch of keys from the given range at `block`.
	/// Returns the keys and whether the batch was full (indicating more keys may exist).
	///
	/// Note: This method handles connection errors by indicating a restart is needed.
	/// The caller should handle reconnection logic.
	async fn rpc_get_keys_single_batch(
		&self,
		range: KeyRange,
		block: B::Hash,
		worker_index: usize,
		client: &WsClient,
	) -> Result<(Vec<StorageKey>, bool)> {
		// Fetch a single page of keys with retry logic
		// Note: The retry logic in get_keys_single_page handles transient errors,
		// but connection errors need to be propagated up for reconnection
		let mut page = self
			.get_keys_single_page_with_client(
				client,
				Some(range.prefix.clone()),
				Some(range.start_key.clone()),
				block,
			)
			.await?;

		// Avoid duplicated keys across workloads - filter out keys beyond our range
		if let (Some(last), Some(end)) = (page.last(), &range.end_key) {
			if last >= end {
				page.retain(|key| key < end);
			}
		}

		let page_len = page.len();
		let is_full_batch = page_len == Self::DEFAULT_KEY_DOWNLOAD_PAGE as usize;

		debug!(
			target: LOG_TARGET,
			"Worker {worker_index}: fetched {} keys from range, full_batch={}",
			page_len,
			is_full_batch
		);

		Ok((page, is_full_batch))
	}

	/// Subdivide the key space AFTER the last_key into up to 16 new ranges.
	/// Uses the same subdivision logic as gen_key_ranges but applies it to the remaining space.
	fn subdivide_remaining_range(
		last_key: &StorageKey,
		end_key: Option<&StorageKey>,
		prefix: &StorageKey,
	) -> Vec<KeyRange> {
		// Create a synthetic "start" position one after the last key
		// We need to increment the last key by 1 to get the true start of remaining range
		let last_key_bytes = last_key.as_ref();
		let mut remaining_start = last_key_bytes.to_vec();

		// Increment the key by 1 (handle overflow properly)
		let mut carry = true;
		for byte in remaining_start.iter_mut().rev() {
			if carry {
				if *byte == 255 {
					*byte = 0;
					// carry remains true
				} else {
					*byte += 1;
					carry = false;
					break;
				}
			}
		}

		// If we still have carry, we've overflowed the entire key space
		if carry {
			return vec![];
		}

		let remaining_start_key = StorageKey(remaining_start.clone());

		// Check if remaining_start is already past the end_key
		if let Some(end) = end_key {
			if &remaining_start_key >= end {
				return vec![];
			}
		}

		// Now subdivide the space from remaining_start to end_key
		// Use similar logic to gen_key_ranges
		let scale = 32usize.saturating_sub(remaining_start.len());

		// If the key is already near max length or the range is small, don't subdivide
		if scale < 2 {
			return vec![KeyRange::new(remaining_start_key, end_key.cloned(), prefix.clone())];
		}

		// Create up to 16 subdivisions
		let chunks = 16;
		let step = 0x10000 / chunks;
		let ext = scale.saturating_sub(2);

		let mut ranges = Vec::new();
		for i in 0..chunks {
			let mut start = remaining_start.clone();
			let chunk_start = i * step;
			start.extend(vec![(chunk_start >> 8) as u8, (chunk_start & 0xff) as u8]);
			start.extend(vec![0; ext]);

			let start_key = StorageKey(start);

			// Skip if this range starts before our actual start point
			if start_key < remaining_start_key {
				continue;
			}

			// Skip if this range starts at or after the end
			if let Some(end) = end_key {
				if &start_key >= end {
					break;
				}
			}

			// Compute end for this chunk
			let chunk_end_key = if i < chunks - 1 {
				let mut end = remaining_start.clone();
				let chunk_end = (i + 1) * step;
				end.extend(vec![(chunk_end >> 8) as u8, (chunk_end & 0xff) as u8]);
				end.extend(vec![0; ext]);
				let computed_end = StorageKey(end);

				// Use minimum of computed end and actual range end
				Some(match end_key {
					Some(actual_end) if &computed_end > actual_end => actual_end.clone(),
					_ => computed_end,
				})
			} else {
				end_key.cloned()
			};

			ranges.push(KeyRange::new(start_key, chunk_end_key, prefix.clone()));
		}

		ranges
	}

	/// Fetches storage data from a node using a dynamic batch size.
	///
	/// This function adjusts the batch size on the fly to help prevent overwhelming the node with
	/// large batch requests, and stay within request size limits enforced by the node.
	///
	/// # Arguments
	///
	/// * `client` - An `Arc` wrapped `HttpClient` used for making the requests.
	/// * `payloads` - A vector of tuples containing a JSONRPC method name and `ArrayParams`
	///
	/// # Returns
	///
	/// Returns a `Result` with a vector of `Option<StorageData>`, where each element corresponds to
	/// the storage data for the given method and parameters. The result will be an `Err` with a
	/// `String` error message if the request fails.
	///
	/// # Errors
	///
	/// This function will return an error if:
	/// * The batch request fails and the batch size is less than 2.
	/// * There are invalid batch params.
	/// * There is an error in the batch response.
	///
	/// # Example
	///
	/// ```ignore
	/// use your_crate::{get_storage_data_dynamic_batch_size, HttpClient, ArrayParams};
	/// use std::sync::Arc;
	///
	/// async fn example() {
	///     let client = HttpClient::new();
	///     let payloads = vec![
	///         ("some_method".to_string(), ArrayParams::new(vec![])),
	///         ("another_method".to_string(), ArrayParams::new(vec![])),
	///     ];
	///     let initial_batch_size = 10;
	///
	///     let storage_data = get_storage_data_dynamic_batch_size(client, payloads, batch_size).await;
	///     match storage_data {
	///         Ok(data) => println!("Storage data: {:?}", data),
	///         Err(e) => eprintln!("Error fetching storage data: {}", e),
	///     }
	/// }
	/// ```
	async fn get_storage_data_dynamic_batch_size(
		conn_manager: &ConnectionManager,
		worker_index: usize,
		payloads: Vec<(String, ArrayParams)>,
		bar: &ProgressBar,
	) -> Result<Vec<Option<StorageData>>, String> {
		let mut all_data: Vec<Option<StorageData>> = vec![];
		let mut start_index = 0;
		let mut retries = 0usize;
		let mut batch_size = Self::INITIAL_BATCH_SIZE;
		let total_payloads = payloads.len();

		while start_index < total_payloads {
			debug!(
				target: LOG_TARGET,
				"Value worker {worker_index}: Remaining payloads: {} Batch request size: {batch_size}",
				total_payloads - start_index,
			);

			let end_index = usize::min(start_index + batch_size, total_payloads);
			let page = &payloads[start_index..end_index];

			// Build the batch request
			let mut batch = BatchRequestBuilder::new();
			for (method, params) in page.iter() {
				batch
					.insert(method, params.clone())
					.map_err(|_| "Invalid batch method and/or params")?;
			}

			// Get client for this worker
			let client = conn_manager.get_client(worker_index).await;

			let request_started = Instant::now();
			let batch_response = match client.batch_request::<Option<StorageData>>(batch).await {
				Ok(batch_response) => {
					retries = 0;
					batch_response
				},
				Err(e) => {
					// Check if this looks like a connection error that can be fixed by recreating
					// the client
					let error_msg = e.to_string().to_lowercase();
					let is_connection_error = error_msg.contains("connection closed") ||
						error_msg.contains("connection reset") ||
						error_msg.contains("broken pipe") ||
						error_msg.contains("connection refused") ||
						error_msg.contains("background task closed") ||
						error_msg.contains("restart required");

					if is_connection_error {
						warn!(
							target: LOG_TARGET,
							"Value worker {worker_index}: Connection error detected, attempting to recreate client: {e}"
						);

						// Try to recreate the WebSocket client
						if let Err(recreate_err) = conn_manager.recreate_client(worker_index).await
						{
							error!(
								target: LOG_TARGET,
								"Value worker {worker_index}: Failed to recreate client: {recreate_err:?}"
							);
						} else {
							debug!(
								target: LOG_TARGET,
								"Value worker {worker_index}: Successfully recreated client"
							);
						}

						// Reset retries and batch size after connection recreation
						// Use a small batch size to avoid overwhelming the new connection
						retries = 0;
						batch_size = 1;
						continue;
					}

					if retries > Self::MAX_RETRIES {
						return Err(e.to_string())
					}

					retries += 1;
					let failure_log = format!(
						"Value worker {worker_index}: Batch request failed ({retries}/{} retries). Error: {e}",
						Self::MAX_RETRIES
					);
					// after 2 subsequent failures something very wrong is happening. log a warning
					// and reset the batch size down to 1.
					if retries >= 2 {
						warn!("{failure_log}");
						batch_size = 1;
					} else {
						debug!("{failure_log}");
						// Decrease batch size by DECREASE_FACTOR
						batch_size =
							(batch_size as f32 * Self::BATCH_SIZE_DECREASE_FACTOR) as usize;
					}
					continue
				},
			};

			let request_duration = request_started.elapsed();
			batch_size = if request_duration > Self::REQUEST_DURATION_TARGET {
				// Decrease batch size
				max(1, (batch_size as f32 * Self::BATCH_SIZE_DECREASE_FACTOR) as usize)
			} else {
				// Increase batch size, but not more than the remaining total payloads to process
				min(
					total_payloads - start_index,
					max(
						batch_size + 1,
						(batch_size as f32 * Self::BATCH_SIZE_INCREASE_FACTOR) as usize,
					),
				)
			};

			debug!(
				target: LOG_TARGET,
				"Value worker {worker_index}: Request duration: {request_duration:?} Target duration: {:?} Last batch size: {} Next batch size: {batch_size}",
				Self::REQUEST_DURATION_TARGET,
				end_index - start_index,
			);

			let batch_response_len = batch_response.len();
			for item in batch_response.into_iter() {
				match item {
					Ok(x) => all_data.push(x),
					Err(e) => return Err(e.message().to_string()),
				}
			}
			bar.inc(batch_response_len as u64);

			// Update the start index for the next iteration
			start_index = end_index;
		}

		Ok(all_data)
	}

	/// Synonym of `getPairs` that uses paged queries to first get the keys, and then
	/// map them to values one by one.
	///
	/// This can work with public nodes. But, expect it to be darn slow.
	pub(crate) async fn rpc_get_pairs(
		&self,
		prefix: StorageKey,
		at: B::Hash,
		pending_ext: &mut TestExternalities<HashingFor<B>>,
	) -> Result<Vec<KeyValue>> {
		let keys = logging::with_elapsed_async(
			|| async {
				// TODO: We could start downloading when having collected the first batch of keys.
				// https://github.com/paritytech/polkadot-sdk/issues/2494
				let keys = self
					.rpc_get_keys_parallel(&prefix, at, Self::PARALLEL_REQUESTS)
					.await?
					.into_iter()
					.collect::<Vec<_>>();

				Ok(keys)
			},
			"Scraping keys...",
			|keys| format!("Found {} keys", keys.len()),
		)
		.await?;

		if keys.is_empty() {
			return Ok(Default::default())
		}

		// Create ConnectionManager for value fetching across multiple providers
		let config = self.as_online();
		let conn_manager = ConnectionManager::new(config.transports.clone())?;

		let payloads = keys
			.iter()
			.map(|key| ("state_getStorage".to_string(), rpc_params!(key, at)))
			.collect::<Vec<_>>();

		let bar = ProgressBar::new(payloads.len() as u64);
		bar.enable_steady_tick(Duration::from_secs(1));
		bar.set_message("Downloading key values".to_string());
		bar.set_style(
			ProgressStyle::with_template(
				"[{elapsed_precise}] {msg} {per_sec} [{wide_bar}] {pos}/{len} ({eta})",
			)
			.unwrap()
			.progress_chars("=>-"),
		);

		// Distribute work across PARALLEL_REQUESTS workers, each using a different provider
		let payloads_chunked = payloads.chunks((payloads.len() / Self::PARALLEL_REQUESTS).max(1));
		let requests = payloads_chunked.enumerate().map(|(worker_index, payload_chunk)| {
			Self::get_storage_data_dynamic_batch_size(
				&conn_manager,
				worker_index,
				payload_chunk.to_vec(),
				&bar,
			)
		});

		// Execute the requests and move the Result outside.
		let storage_data_result: Result<Vec<_>, _> =
			futures::future::join_all(requests).await.into_iter().collect();

		// Handle the Result.
		let storage_data = match storage_data_result {
			Ok(storage_data) => storage_data.into_iter().flatten().collect::<Vec<_>>(),
			Err(e) => {
				error!(target: LOG_TARGET, "Error while getting storage data: {e}");
				return Err("Error while getting storage data")
			},
		};
		bar.finish_with_message("✅ Downloaded key values");
		println!();

		// Check if we got responses for all submitted requests.
		assert_eq!(keys.len(), storage_data.len());

		let key_values = keys
			.iter()
			.zip(storage_data)
			.map(|(key, maybe_value)| match maybe_value {
				Some(data) => (key.clone(), data),
				None => {
					warn!(target: LOG_TARGET, "key {key:?} had none corresponding value.");
					let data = StorageData(vec![]);
					(key.clone(), data)
				},
			})
			.collect::<Vec<_>>();

		logging::with_elapsed(
			|| {
				pending_ext.batch_insert(key_values.clone().into_iter().filter_map(|(k, v)| {
					// Don't insert the child keys here, they need to be inserted separately with
					// all their data in the load_child_remote function.
					match is_default_child_storage_key(&k.0) {
						true => None,
						false => Some((k.0, v.0)),
					}
				}));

				Ok(())
			},
			"Inserting keys into DB...",
			|_| "Inserted keys into DB".into(),
		)
		.expect("must succeed; qed");

		Ok(key_values)
	}

	/// Get the values corresponding to `child_keys` at the given `prefixed_top_key`.
	pub(crate) async fn rpc_child_get_storage_paged(
		conn_manager: &ConnectionManager,
		worker_index: usize,
		prefixed_top_key: &StorageKey,
		child_keys: Vec<StorageKey>,
		at: B::Hash,
	) -> Result<Vec<KeyValue>> {
		let child_keys_len = child_keys.len();

		let payloads = child_keys
			.iter()
			.map(|key| {
				(
					"childstate_getStorage".to_string(),
					rpc_params![
						PrefixedStorageKey::new(prefixed_top_key.as_ref().to_vec()),
						key,
						at
					],
				)
			})
			.collect::<Vec<_>>();

		let bar = ProgressBar::new(payloads.len() as u64);
		let storage_data = match Self::get_storage_data_dynamic_batch_size(
			conn_manager,
			worker_index,
			payloads,
			&bar,
		)
		.await
		{
			Ok(storage_data) => storage_data,
			Err(e) => {
				error!(target: LOG_TARGET, "batch processing failed: {e:?}");
				return Err("batch processing failed")
			},
		};

		assert_eq!(child_keys_len, storage_data.len());

		Ok(child_keys
			.iter()
			.zip(storage_data)
			.map(|(key, maybe_value)| match maybe_value {
				Some(v) => (key.clone(), v),
				None => {
					warn!(target: LOG_TARGET, "key {key:?} had no corresponding value.");
					(key.clone(), StorageData(vec![]))
				},
			})
			.collect::<Vec<_>>())
	}

	pub(crate) async fn rpc_child_get_keys(
		client: &WsClient,
		prefixed_top_key: &StorageKey,
		child_prefix: StorageKey,
		at: B::Hash,
	) -> Result<Vec<StorageKey>> {
		let retry_strategy =
			FixedInterval::new(Self::KEYS_PAGE_RETRY_INTERVAL).take(Self::MAX_RETRIES);
		let mut all_child_keys = Vec::new();
		let mut start_key = None;

		loop {
			let get_child_keys_closure = || {
				let top_key = PrefixedStorageKey::new(prefixed_top_key.0.clone());
				substrate_rpc_client::ChildStateApi::storage_keys_paged(
					client,
					top_key,
					Some(child_prefix.clone()),
					Self::DEFAULT_KEY_DOWNLOAD_PAGE,
					start_key.clone(),
					Some(at),
				)
			};

			let child_keys = Retry::spawn(retry_strategy.clone(), get_child_keys_closure)
				.await
				.map_err(|e| {
					error!(target: LOG_TARGET, "Error = {e:?}");
					"rpc child_get_keys failed."
				})?;

			let keys_count = child_keys.len();
			if keys_count == 0 {
				break;
			}

			start_key = child_keys.last().cloned();
			all_child_keys.extend(child_keys);

			if keys_count < Self::DEFAULT_KEY_DOWNLOAD_PAGE as usize {
				break;
			}
		}

		debug!(
			target: LOG_TARGET,
			"[thread = {:?}] scraped {} child-keys of the child-bearing top key: {}",
			std::thread::current().id(),
			all_child_keys.len(),
			HexDisplay::from(prefixed_top_key)
		);

		Ok(all_child_keys)
	}
}

impl<B: BlockT> Builder<B>
where
	B::Hash: DeserializeOwned,
	B::Header: DeserializeOwned,
{
	/// Load all of the child keys from the remote config, given the already scraped list of top key
	/// pairs.
	///
	/// `top_kv` need not be only child-bearing top keys. It should be all of the top keys that are
	/// included thus far.
	///
	/// This function concurrently populates `pending_ext`. the return value is only for writing to
	/// cache, we can also optimize further.
	async fn load_child_remote(
		&self,
		top_kv: &[KeyValue],
		pending_ext: &mut TestExternalities<HashingFor<B>>,
	) -> Result<ChildKeyValues> {
		let child_roots = top_kv
			.iter()
			.filter(|(k, _)| is_default_child_storage_key(k.as_ref()))
			.map(|(k, _)| k.clone())
			.collect::<Vec<_>>();

		if child_roots.is_empty() {
			info!(target: LOG_TARGET, "👩‍👦 no child roots found to scrape");
			return Ok(Default::default())
		}

		info!(
			target: LOG_TARGET,
			"👩‍👦 scraping child-tree data from {} top keys",
			child_roots.len(),
		);

		let at = self.as_online().at_expected();

		let config = self.as_online();
		let conn_manager = ConnectionManager::new(config.transports.clone())?;
		let client = self.as_online().rpc_client();
		let mut child_kv = vec![];
		for (worker_index, prefixed_top_key) in child_roots.iter().enumerate() {
			let child_keys =
				Self::rpc_child_get_keys(client, &prefixed_top_key, StorageKey(vec![]), at).await?;

			let child_kv_inner = Self::rpc_child_get_storage_paged(
				&conn_manager,
				worker_index,
				&prefixed_top_key,
				child_keys,
				at,
			)
			.await?;

			let prefixed_top_key = PrefixedStorageKey::new(prefixed_top_key.clone().0);
			let un_prefixed = match ChildType::from_prefixed_key(&prefixed_top_key) {
				Some((ChildType::ParentKeyId, storage_key)) => storage_key,
				None => {
					error!(target: LOG_TARGET, "invalid key: {prefixed_top_key:?}");
					return Err("Invalid child key")
				},
			};

			let info = ChildInfo::new_default(un_prefixed);
			let key_values =
				child_kv_inner.iter().cloned().map(|(k, v)| (k.0, v.0)).collect::<Vec<_>>();
			child_kv.push((info.clone(), child_kv_inner));
			for (k, v) in key_values {
				pending_ext.insert_child(info.clone(), k, v);
			}
		}

		Ok(child_kv)
	}

	/// Build `Self` from a network node denoted by `uri`.
	///
	/// This function concurrently populates `pending_ext`. the return value is only for writing to
	/// cache, we can also optimize further.
	async fn load_top_remote(
		&self,
		pending_ext: &mut TestExternalities<HashingFor<B>>,
	) -> Result<TopKeyValues> {
		let config = self.as_online();
		let at = self
			.as_online()
			.at
			.expect("online config must be initialized by this point; qed.");
		info!(target: LOG_TARGET, "scraping key-pairs from remote at block height {at:?}");

		let mut keys_and_values = Vec::new();
		for prefix in &config.hashed_prefixes {
			let now = std::time::Instant::now();
			let additional_key_values =
				self.rpc_get_pairs(StorageKey(prefix.to_vec()), at, pending_ext).await?;
			let elapsed = now.elapsed();
			info!(
				target: LOG_TARGET,
				"adding data for hashed prefix: {:?}, took {:.2}s",
				HexDisplay::from(prefix),
				elapsed.as_secs_f32()
			);
			keys_and_values.extend(additional_key_values);
		}

		for key in &config.hashed_keys {
			let key = StorageKey(key.to_vec());
			info!(
				target: LOG_TARGET,
				"adding data for hashed key: {:?}",
				HexDisplay::from(&key)
			);
			match self.rpc_get_storage(key.clone(), Some(at)).await? {
				Some(value) => {
					pending_ext.insert(key.clone().0, value.clone().0);
					keys_and_values.push((key, value));
				},
				None => {
					warn!(
						target: LOG_TARGET,
						"no data found for hashed key: {:?}",
						HexDisplay::from(&key)
					);
				},
			}
		}

		Ok(keys_and_values)
	}

	/// The entry point of execution, if `mode` is online.
	///
	/// initializes the remote client in `transport`, and sets the `at` field, if not specified.
	async fn init_remote_client(&mut self) -> Result<()> {
		// First, create all transport clients from URIs.
		let online_config = self.as_online_mut();
		let mut transports = Vec::new();
		for uri in &online_config.transport_uris {
			transports.push(Transport::new(uri.clone()).await?);
		}
		online_config.transports = transports;

		// Then, if `at` is not set, set it.
		if self.as_online().at.is_none() {
			let at = self.rpc_get_head().await?;
			info!(
				target: LOG_TARGET,
				"since no at is provided, setting it to latest finalized head, {at:?}",
			);
			self.as_online_mut().at = Some(at);
		}

		// Then, a few transformation that we want to perform in the online config:
		let online_config = self.as_online_mut();
		online_config.pallets.iter().for_each(|p| {
			online_config
				.hashed_prefixes
				.push(sp_crypto_hashing::twox_128(p.as_bytes()).to_vec())
		});

		if online_config.child_trie {
			online_config.hashed_prefixes.push(DEFAULT_CHILD_STORAGE_KEY_PREFIX.to_vec());
		}

		// Finally, if by now, we have put any limitations on prefixes that we are interested in, we
		// download everything.
		if online_config
			.hashed_prefixes
			.iter()
			.filter(|p| *p != DEFAULT_CHILD_STORAGE_KEY_PREFIX)
			.count() == 0
		{
			info!(
				target: LOG_TARGET,
				"since no prefix is filtered, the data for all pallets will be downloaded"
			);
			online_config.hashed_prefixes.push(vec![]);
		}

		Ok(())
	}

	async fn load_header(&self) -> Result<B::Header> {
		let client = self.as_online().rpc_client();
		let at = self.as_online().at_expected();
		let retry_strategy =
			FixedInterval::new(Self::KEYS_PAGE_RETRY_INTERVAL).take(Self::MAX_RETRIES);
		let get_header_closure = || ChainApi::<(), _, B::Header, ()>::header(client, Some(at));
		Retry::spawn(retry_strategy, get_header_closure)
			.await
			.map_err(|_| "Failed to fetch header for block from network")?
			.ok_or("Network returned None block header")
	}

	/// Load the data from a remote server. The main code path is calling into `load_top_remote` and
	/// `load_child_remote`.
	///
	/// Must be called after `init_remote_client`.
	async fn load_remote_and_maybe_save(&mut self) -> Result<TestExternalities<HashingFor<B>>> {
		let state_version =
			StateApi::<B::Hash>::runtime_version(self.as_online().rpc_client(), None)
				.await
				.map_err(|e| {
					error!(target: LOG_TARGET, "Error = {e:?}");
					"rpc runtime_version failed."
				})
				.map(|v| v.state_version())?;
		let mut pending_ext = TestExternalities::new_with_code_and_state(
			Default::default(),
			Default::default(),
			self.overwrite_state_version.unwrap_or(state_version),
		);

		// Load data from the remote into `pending_ext`.
		let top_kv = self.load_top_remote(&mut pending_ext).await?;
		self.load_child_remote(&top_kv, &mut pending_ext).await?;

		// If we need to save a snapshot, save the raw storage and root hash to the snapshot.
		if let Some(path) = self.as_online().state_snapshot.clone().map(|c| c.path) {
			let (raw_storage, storage_root) = pending_ext.into_raw_snapshot();
			let snapshot = Snapshot::<B>::new(
				state_version,
				raw_storage.clone(),
				storage_root,
				self.load_header().await?,
			);
			let encoded = snapshot.encode();
			info!(
				target: LOG_TARGET,
				"writing snapshot of {} bytes to {path:?}",
				encoded.len(),
			);
			std::fs::write(path, encoded).map_err(|_| "fs::write failed")?;

			// pending_ext was consumed when creating the snapshot, need to reinitailize it
			return Ok(TestExternalities::from_raw_snapshot(
				raw_storage,
				storage_root,
				self.overwrite_state_version.unwrap_or(state_version),
			))
		}

		Ok(pending_ext)
	}

	async fn do_load_remote(&mut self) -> Result<RemoteExternalities<B>> {
		self.init_remote_client().await?;
		let inner_ext = self.load_remote_and_maybe_save().await?;
		Ok(RemoteExternalities { header: self.load_header().await?, inner_ext })
	}

	fn do_load_offline(&mut self, config: OfflineConfig) -> Result<RemoteExternalities<B>> {
		let (header, inner_ext) = logging::with_elapsed(
			|| {
				info!(target: LOG_TARGET, "Loading snapshot from {:?}", &config.state_snapshot.path);

				let Snapshot { header, state_version, raw_storage, storage_root, .. } =
					Snapshot::<B>::load(&config.state_snapshot.path)?;
				let inner_ext = TestExternalities::from_raw_snapshot(
					raw_storage,
					storage_root,
					self.overwrite_state_version.unwrap_or(state_version),
				);

				Ok((header, inner_ext))
			},
			"Loading snapshot...",
			|_| "Loaded snapshot".into(),
		)?;

		Ok(RemoteExternalities { inner_ext, header })
	}

	pub(crate) async fn pre_build(mut self) -> Result<RemoteExternalities<B>> {
		let mut ext = match self.mode.clone() {
			Mode::Offline(config) => self.do_load_offline(config)?,
			Mode::Online(_) => self.do_load_remote().await?,
			Mode::OfflineOrElseOnline(offline_config, _) => {
				match self.do_load_offline(offline_config) {
					Ok(x) => x,
					Err(_) => self.do_load_remote().await?,
				}
			},
		};

		// inject manual key values.
		if !self.hashed_key_values.is_empty() {
			info!(
				target: LOG_TARGET,
				"extending externalities with {} manually injected key-values",
				self.hashed_key_values.len()
			);
			ext.batch_insert(self.hashed_key_values.into_iter().map(|(k, v)| (k.0, v.0)));
		}

		// exclude manual key values.
		if !self.hashed_blacklist.is_empty() {
			info!(
				target: LOG_TARGET,
				"excluding externalities from {} keys",
				self.hashed_blacklist.len()
			);
			for k in self.hashed_blacklist {
				ext.execute_with(|| sp_io::storage::clear(&k));
			}
		}

		Ok(ext)
	}
}

// Public methods
impl<B: BlockT> Builder<B>
where
	B::Hash: DeserializeOwned,
	B::Header: DeserializeOwned,
{
	/// Create a new builder.
	pub fn new() -> Self {
		Default::default()
	}

	/// Inject a manual list of key and values to the storage.
	pub fn inject_hashed_key_value(mut self, injections: Vec<KeyValue>) -> Self {
		for i in injections {
			self.hashed_key_values.push(i.clone());
		}
		self
	}

	/// Blacklist this hashed key from the final externalities. This is treated as-is, and should be
	/// pre-hashed.
	pub fn blacklist_hashed_key(mut self, hashed: &[u8]) -> Self {
		self.hashed_blacklist.push(hashed.to_vec());
		self
	}

	/// Configure a state snapshot to be used.
	pub fn mode(mut self, mode: Mode<B::Hash>) -> Self {
		self.mode = mode;
		self
	}

	/// The state version to use.
	pub fn overwrite_state_version(mut self, version: StateVersion) -> Self {
		self.overwrite_state_version = Some(version);
		self
	}

	pub async fn build(self) -> Result<RemoteExternalities<B>> {
		let mut ext = self.pre_build().await?;
		ext.commit_all().unwrap();

		info!(
			target: LOG_TARGET,
			"initialized state externalities with storage root {:?} and state_version {:?}",
			ext.as_backend().root(),
			ext.state_version
		);

		Ok(ext)
	}
}

#[cfg(test)]
mod test_prelude {
	pub(crate) use super::*;
	pub(crate) use sp_runtime::testing::{Block as RawBlock, MockCallU64};
	pub(crate) type UncheckedXt = sp_runtime::testing::TestXt<MockCallU64, ()>;
	pub(crate) type Block = RawBlock<UncheckedXt>;

	pub(crate) fn init_logger() {
		sp_tracing::try_init_simple();
	}
}

#[cfg(test)]
mod tests {
	use super::test_prelude::*;

	#[tokio::test]
	async fn can_load_state_snapshot() {
		init_logger();
		Builder::<Block>::new()
			.mode(Mode::Offline(OfflineConfig {
				state_snapshot: SnapshotConfig::new("test_data/test.snap"),
			}))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});
	}

	#[tokio::test]
	async fn can_exclude_from_snapshot() {
		init_logger();

		// get the first key from the snapshot file.
		let some_key = Builder::<Block>::new()
			.mode(Mode::Offline(OfflineConfig {
				state_snapshot: SnapshotConfig::new("test_data/test.snap"),
			}))
			.build()
			.await
			.expect("Can't read state snapshot file")
			.execute_with(|| {
				let key =
					sp_io::storage::next_key(&[]).expect("some key must exist in the snapshot");
				assert!(sp_io::storage::get(&key).is_some());
				key
			});

		Builder::<Block>::new()
			.mode(Mode::Offline(OfflineConfig {
				state_snapshot: SnapshotConfig::new("test_data/test.snap"),
			}))
			.blacklist_hashed_key(&some_key)
			.build()
			.await
			.expect("Can't read state snapshot file")
			.execute_with(|| assert!(sp_io::storage::get(&some_key).is_none()));
	}
}

#[cfg(all(test, feature = "remote-test"))]
mod remote_tests {
	use super::test_prelude::*;
	use std::{env, os::unix::fs::MetadataExt};

	fn endpoint() -> String {
		env::var("TEST_WS").unwrap_or_else(|_| DEFAULT_HTTP_ENDPOINT.to_string())
	}

	#[tokio::test]
	async fn state_version_is_kept_and_can_be_altered() {
		const CACHE: &'static str = "state_version_is_kept_and_can_be_altered";
		init_logger();

		// first, build a snapshot.
		let ext = Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				pallets: vec!["Proxy".to_owned()],
				child_trie: false,
				state_snapshot: Some(SnapshotConfig::new(CACHE)),
				..Default::default()
			}))
			.build()
			.await
			.unwrap();

		// now re-create the same snapshot.
		let cached_ext = Builder::<Block>::new()
			.mode(Mode::Offline(OfflineConfig { state_snapshot: SnapshotConfig::new(CACHE) }))
			.build()
			.await
			.unwrap();

		assert_eq!(ext.state_version, cached_ext.state_version);

		// now overwrite it
		let other = match ext.state_version {
			StateVersion::V0 => StateVersion::V1,
			StateVersion::V1 => StateVersion::V0,
		};
		let cached_ext = Builder::<Block>::new()
			.mode(Mode::Offline(OfflineConfig { state_snapshot: SnapshotConfig::new(CACHE) }))
			.overwrite_state_version(other)
			.build()
			.await
			.unwrap();

		assert_eq!(cached_ext.state_version, other);
	}

	#[tokio::test]
	async fn snapshot_block_hash_works() {
		const CACHE: &'static str = "snapshot_block_hash_works";
		init_logger();

		// first, build a snapshot.
		let ext = Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				pallets: vec!["Proxy".to_owned()],
				child_trie: false,
				state_snapshot: Some(SnapshotConfig::new(CACHE)),
				..Default::default()
			}))
			.build()
			.await
			.unwrap();

		// now re-create the same snapshot.
		let cached_ext = Builder::<Block>::new()
			.mode(Mode::Offline(OfflineConfig { state_snapshot: SnapshotConfig::new(CACHE) }))
			.build()
			.await
			.unwrap();

		assert_eq!(ext.header.hash(), cached_ext.header.hash());
	}

	#[tokio::test]
	async fn child_keys_are_loaded() {
		const CACHE: &'static str = "snapshot_retains_storage";
		init_logger();

		// create an ext with children keys
		let mut child_ext = Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				pallets: vec!["Proxy".to_owned()],
				child_trie: true,
				state_snapshot: Some(SnapshotConfig::new(CACHE)),
				..Default::default()
			}))
			.build()
			.await
			.unwrap();

		// create an ext without children keys
		let mut ext = Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				pallets: vec!["Proxy".to_owned()],
				child_trie: false,
				state_snapshot: Some(SnapshotConfig::new(CACHE)),
				..Default::default()
			}))
			.build()
			.await
			.unwrap();

		// there should be more keys in the child ext.
		assert!(
			child_ext.as_backend().backend_storage().keys().len() >
				ext.as_backend().backend_storage().keys().len()
		);
	}

	#[tokio::test]
	async fn offline_else_online_works() {
		const CACHE: &'static str = "offline_else_online_works_data";
		init_logger();
		// this shows that in the second run, we use the remote and create a snapshot.
		Builder::<Block>::new()
			.mode(Mode::OfflineOrElseOnline(
				OfflineConfig { state_snapshot: SnapshotConfig::new(CACHE) },
				OnlineConfig {
					transport_uris: vec![endpoint().clone()],
					pallets: vec!["Proxy".to_owned()],
					child_trie: false,
					state_snapshot: Some(SnapshotConfig::new(CACHE)),
					..Default::default()
				},
			))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});

		// this shows that in the second run, we are not using the remote
		Builder::<Block>::new()
			.mode(Mode::OfflineOrElseOnline(
				OfflineConfig { state_snapshot: SnapshotConfig::new(CACHE) },
				OnlineConfig {
					transport_uris: vec!["ws://non-existent:666".to_owned()],
					..Default::default()
				},
			))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});

		let to_delete = std::fs::read_dir(Path::new("."))
			.unwrap()
			.into_iter()
			.map(|d| d.unwrap())
			.filter(|p| p.path().file_name().unwrap_or_default() == CACHE)
			.collect::<Vec<_>>();

		assert!(to_delete.len() == 1);
		std::fs::remove_file(to_delete[0].path()).unwrap();
	}

	#[tokio::test]
	async fn can_build_one_small_pallet() {
		init_logger();
		Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				pallets: vec!["Proxy".to_owned()],
				child_trie: false,
				..Default::default()
			}))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});
	}

	#[tokio::test]
	async fn can_build_few_pallet() {
		init_logger();
		Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				pallets: vec!["Proxy".to_owned(), "Multisig".to_owned()],
				child_trie: false,
				..Default::default()
			}))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});
	}

	#[tokio::test(flavor = "multi_thread")]
	async fn can_create_snapshot() {
		const CACHE: &'static str = "can_create_snapshot";
		init_logger();

		Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				state_snapshot: Some(SnapshotConfig::new(CACHE)),
				pallets: vec!["Proxy".to_owned()],
				child_trie: false,
				..Default::default()
			}))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});

		let to_delete = std::fs::read_dir(Path::new("."))
			.unwrap()
			.into_iter()
			.map(|d| d.unwrap())
			.filter(|p| p.path().file_name().unwrap_or_default() == CACHE)
			.collect::<Vec<_>>();

		assert!(to_delete.len() == 1);
		let to_delete = to_delete.first().unwrap();
		assert!(std::fs::metadata(to_delete.path()).unwrap().size() > 1);
		std::fs::remove_file(to_delete.path()).unwrap();
	}

	#[tokio::test]
	async fn can_create_child_snapshot() {
		const CACHE: &'static str = "can_create_child_snapshot";
		init_logger();
		Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				state_snapshot: Some(SnapshotConfig::new(CACHE)),
				pallets: vec!["Crowdloan".to_owned()],
				child_trie: true,
				..Default::default()
			}))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});

		let to_delete = std::fs::read_dir(Path::new("."))
			.unwrap()
			.into_iter()
			.map(|d| d.unwrap())
			.filter(|p| p.path().file_name().unwrap_or_default() == CACHE)
			.collect::<Vec<_>>();

		assert!(to_delete.len() == 1);
		let to_delete = to_delete.first().unwrap();
		assert!(std::fs::metadata(to_delete.path()).unwrap().size() > 1);
		std::fs::remove_file(to_delete.path()).unwrap();
	}

	#[tokio::test]
	async fn can_build_big_pallet() {
		if std::option_env!("TEST_WS").is_none() {
			return
		}
		init_logger();
		Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				pallets: vec!["Staking".to_owned()],
				child_trie: false,
				..Default::default()
			}))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});
	}

	#[tokio::test]
	async fn can_fetch_all() {
		if std::option_env!("TEST_WS").is_none() {
			return
		}
		init_logger();
		Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: vec![endpoint().clone()],
				..Default::default()
			}))
			.build()
			.await
			.unwrap()
			.execute_with(|| {});
	}

	#[tokio::test]
	async fn can_fetch_in_parallel() {
		init_logger();

		let mut builder = Builder::<Block>::new().mode(Mode::Online(OnlineConfig {
			transport_uris: vec![endpoint().clone()],
			..Default::default()
		}));
		builder.init_remote_client().await.unwrap();

		let at = builder.as_online().at.unwrap();

		// Test with a specific prefix
		let prefix = StorageKey(vec![13]);
		let para = builder.rpc_get_keys_parallel(&prefix, at, 4).await.unwrap();
		assert!(!para.is_empty(), "Should fetch some keys with prefix");

		// Test with empty prefix (all keys)
		let prefix = StorageKey(vec![]);
		let para = builder.rpc_get_keys_parallel(&prefix, at, 8).await.unwrap();
		assert!(!para.is_empty(), "Should fetch some keys with empty prefix");
	}

	#[tokio::test]
	#[ignore] // This test takes a long time, run with --ignored
	async fn bridge_hub_polkadot_storage_root_matches() {
		init_logger();

		// Use multiple RPC providers for load distribution
		let endpoints = vec![
			"wss://bridge-hub-polkadot-rpc.n.dwellir.com",
			"wss://sys.ibp.network/bridgehub-polkadot",
			"wss://bridgehub-polkadot.api.onfinality.io/public",
			"wss://dot-rpc.stakeworld.io/bridgehub",
		];

		info!(target: LOG_TARGET, "Connecting to Bridge Hub Polkadot using {} RPC providers", endpoints.len());

		let mut ext = Builder::<Block>::new()
			.mode(Mode::Online(OnlineConfig {
				transport_uris: endpoints.into_iter().map(|e| e.to_owned()).collect(),
				child_trie: true,
				..Default::default()
			}))
			.build()
			.await
			.expect("Failed to build remote externalities");

		// Get the computed storage root from our downloaded state
		let backend = ext.as_backend();
		let computed_root = *backend.root();
		// Get the expected storage root from the block header
		let expected_root = ext.header.state_root;

		info!(
			target: LOG_TARGET,
			"Computed storage root: {:?}",
			computed_root
		);
		info!(
			target: LOG_TARGET,
			"Expected storage root (from header): {:?}",
			expected_root
		);

		// The storage roots must match exactly - this proves we downloaded all keys correctly
		assert_eq!(
			computed_root, expected_root,
			"Storage root mismatch! Computed: {:?}, Expected: {:?}. \
			This indicates that not all keys were fetched or there were duplicates.",
			computed_root, expected_root
		);

		// Verify we actually got some keys
		ext.execute_with(|| {
			let key_count = sp_io::storage::next_key(&[])
				.map(|first_key| {
					let mut count = 1;
					let mut current = first_key;
					while let Some(next) = sp_io::storage::next_key(&current) {
						count += 1;
						current = next;
					}
					count
				})
				.unwrap_or(0);

			info!(target: LOG_TARGET, "Total keys in state: {}", key_count);
			assert!(key_count > 0, "Should have fetched some keys");
		});

		info!(
			target: LOG_TARGET,
			"✅ Storage root verification successful! All keys were fetched correctly."
		);
	}
}
