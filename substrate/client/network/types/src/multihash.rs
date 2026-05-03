// This file is part of Substrate.

// Copyright (C) Parity Technologies (UK) Ltd.
// SPDX-License-Identifier: GPL-3.0-or-later WITH Classpath-exception-2.0

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! [`Multihash`] implemenattion used by substrate. Currently it's a wrapper over
//! multihash used by litep2p, but it can be switched to other implementation if needed.

use litep2p::types::multihash::{
	Error as LiteP2pError, Multihash as LiteP2pMultihashGeneric,
};
use multihash_codetable::MultihashDigest;
use std::fmt::{self, Debug};

const MULTIHASH_SIZE: usize = 64;

const IDENTITY_CODE: u64 = 0x00;
const SHA2_256_CODE: u64 = 0x12;

type LiteP2pMultihash = LiteP2pMultihashGeneric<MULTIHASH_SIZE>;

/// Default [`Multihash`] implementations. Only hashes used by substrate are defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Code {
	/// Identity hasher.
	Identity,
	/// SHA-256 (32-byte hash size).
	Sha2_256,
}

impl Code {
	/// Calculate digest using this [`Code`]'s hashing algorithm.
	pub fn digest(&self, input: &[u8]) -> Multihash {
		match self {
			Code::Identity => Multihash::wrap(IDENTITY_CODE, input)
				.expect("identity digest must fit into the configured multihash size; qed"),
			Code::Sha2_256 => multihash_codetable::Code::Sha2_256.digest(input).into()
		}
	}
}

/// Error generated when converting to [`Code`].
#[derive(Debug, thiserror::Error)]
pub enum Error {
	/// Invalid multihash size.
	#[error("invalid multihash size '{0}'")]
	InvalidSize(u64),
	/// The multihash code is not supported.
	#[error("unsupported multihash code '{0:x}'")]
	UnsupportedCode(u64),
	/// Catch-all for other errors emitted when converting `u64` code to enum or parsing multihash
	/// from bytes. Never generated as of multihash-0.17.0.
	#[error("other error: {0}")]
	Other(Box<dyn std::error::Error + Send + Sync>),
}

impl From<LiteP2pError> for Error {
	fn from(error: LiteP2pError) -> Self {
		Self::Other(Box::new(error))
	}
}

// impl From<Code> for LiteP2pCode {
// 	fn from(code: Code) -> Self {
// 		match code {
// 			Code::Identity => LiteP2pCode::Identity,
// 			Code::Sha2_256 => LiteP2pCode::Sha2_256,
// 		}
// 	}
// }
//
// impl TryFrom<LiteP2pCode> for Code {
// 	type Error = Error;
//
// 	fn try_from(code: LiteP2pCode) -> Result<Self, Self::Error> {
// 		match code {
// 			LiteP2pCode::Identity => Ok(Code::Identity),
// 			LiteP2pCode::Sha2_256 => Ok(Code::Sha2_256),
// 			_ => Err(Error::UnsupportedCode(code.into())),
// 		}
// 	}
// }

impl TryFrom<u64> for Code {
	type Error = Error;

	fn try_from(code: u64) -> Result<Self, Self::Error> {
		match code {
			IDENTITY_CODE => Ok(Code::Identity),
			SHA2_256_CODE => Ok(Code::Sha2_256),
			_ => Err(Error::UnsupportedCode(code)),
		}
	}
}

impl From<Code> for u64 {
	fn from(code: Code) -> Self {
		match code {
			Code::Identity => IDENTITY_CODE,
			Code::Sha2_256 => SHA2_256_CODE,
		}
	}
}

#[derive(Clone, Copy, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct Multihash {
	multihash: LiteP2pMultihash,
}

impl Multihash {
	/// Multihash code.
	pub fn code(&self) -> u64 {
		self.multihash.code()
	}

	/// Multihash digest.
	pub fn digest(&self) -> &[u8] {
		self.multihash.digest()
	}

	/// Wraps the digest in a multihash.
	pub fn wrap(code: u64, input_digest: &[u8]) -> Result<Self, Error> {
		if input_digest.len() > MULTIHASH_SIZE {
			return Err(Error::InvalidSize(input_digest.len() as u64))
		}

		LiteP2pMultihash::wrap(code, input_digest).map(Into::into).map_err(Into::into)
	}

	/// Parses a multihash from bytes.
	///
	/// You need to make sure the passed in bytes have the length of 64.
	pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
		LiteP2pMultihash::from_bytes(bytes).map(Into::into).map_err(Into::into)
	}

	/// Returns the bytes of a multihash.
	pub fn to_bytes(&self) -> Vec<u8> {
		self.multihash.to_bytes()
	}
}

/// Remove extra layer of nestedness by deferring to the wrapped value's [`Debug`].
impl Debug for Multihash {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		Debug::fmt(&self.multihash, f)
	}
}

impl From<LiteP2pMultihash> for Multihash {
	fn from(multihash: LiteP2pMultihash) -> Self {
		Multihash { multihash }
	}
}

impl From<Multihash> for LiteP2pMultihash {
	fn from(multihash: Multihash) -> Self {
		multihash.multihash
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn code_from_u64() {
		assert_eq!(Code::try_from(0x00).unwrap(), Code::Identity);
		assert_eq!(Code::try_from(0x12).unwrap(), Code::Sha2_256);
		assert!(matches!(Code::try_from(0x01).unwrap_err(), Error::UnsupportedCode(0x01)));
	}

	#[test]
	fn code_into_u64() {
		assert_eq!(u64::from(Code::Identity), 0x00);
		assert_eq!(u64::from(Code::Sha2_256), 0x12);
	}

	#[test]
	fn identity_digest() {
		let input = b"substrate";
		let hash = Code::Identity.digest(input);

		assert_eq!(hash.code(), u64::from(Code::Identity));
		assert_eq!(hash.digest(), input);
	}

	#[test]
	fn sha2_256_digest() {
		let hash = Code::Sha2_256.digest(b"substrate");

		assert_eq!(hash.code(), u64::from(Code::Sha2_256));
		assert_eq!(hash.digest().len(), 32);
	}

	#[test]
	fn wrap_rejects_too_large_digest() {
		let digest = [0u8; MULTIHASH_SIZE + 1];

		assert!(matches!(
			Multihash::wrap(u64::from(Code::Identity), &digest),
			Err(Error::InvalidSize(size)) if size == (MULTIHASH_SIZE + 1) as u64
		));
	}
}
