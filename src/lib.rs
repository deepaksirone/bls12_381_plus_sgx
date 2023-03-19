//! # `bls12_381`
//!
//! This crate provides an implementation of the BLS12-381 pairing-friendly elliptic
//! curve construction.
//!
//! * **This implementation has not been reviewed or audited. Use at your own risk.**
//! * This implementation targets Rust `1.36` or later.
//! * This implementation does not require the Rust standard library.
//! * All operations are constant time unless explicitly noted.

#![no_std]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Catch documentation errors caused by code changes.
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![deny(unsafe_code)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
// This lint is described at
// https://rust-lang.github.io/rust-clippy/master/index.html#suspicious_arithmetic_impl
// In our library, some of the arithmetic involving extension fields will necessarily
// involve various binary operators, and so this lint is triggered unnecessarily.
#![allow(clippy::suspicious_arithmetic_impl)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(test)]
#[macro_use]
extern crate std;

pub use elliptic_curve;
pub use ff;
#[cfg(feature = "groups")]
pub use group;

#[cfg(test)]
#[cfg(feature = "groups")]
mod tests;

#[macro_use]
mod util;

/// Notes about how the BLS12-381 elliptic curve is designed, specified
/// and implemented by this library.
pub mod notes {
    pub mod design;
    pub mod serialization;
}

mod scalar;

use elliptic_curve::{Curve, CurveArithmetic};
pub use scalar::Scalar;

#[cfg(all(feature = "groups", not(feature = "expose-fields")))]
mod fp;
#[cfg(feature = "expose-fields")]
pub mod fp;
#[cfg(all(feature = "groups", not(feature = "expose-fields")))]
mod fp2;
#[cfg(feature = "expose-fields")]
pub mod fp2;
#[cfg(feature = "groups")]
mod g1;
#[cfg(feature = "groups")]
mod g2;

#[cfg(feature = "groups")]
pub use g1::{G1Affine, G1Projective};
#[cfg(feature = "expose-fields")]
pub use g1::{G1Compressed, G1Uncompressed};
#[cfg(feature = "groups")]
pub use g2::{G2Affine, G2Projective};
#[cfg(feature = "expose-fields")]
pub use g2::{G2Compressed, G2Uncompressed};

mod fp12;
mod fp6;

// /// An engine for operations generic G1 operations
// #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
// pub struct Bls12381G1;
//
// /// An engine for operations generic G2 operations
// #[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
// pub struct Bls12381G2;
//
// impl CurveArithmetic for Bls12381G1 {
//     type AffinePoint = G1Affine;
//     type ProjectivePoint = G1Projective;
//     type Scalar = Scalar;
// }

// impl Curve for Bls12381G1 {
//     type FieldBytesSize = elliptic_curve::generic_array::typenum::U48;
//     type Uint = elliptic_curve::bigint::U384;
//     const ORDER: Self::Uint = elliptic_curve::bigint::U384::from_be_hex(
//         "1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab"
//     );
// }

// #[cfg(feature = "hashing")]
// impl elliptic_curve::hash2curve::GroupDigest for Bls12381G1 {
//     type FieldElement = fp::Fp;
// }

/// The BLS parameter x for BLS12-381 is -0xd201000000010000
#[cfg(feature = "groups")]
const BLS_X: u64 = 0xd201_0000_0001_0000;
#[cfg(feature = "groups")]
const BLS_X_IS_NEGATIVE: bool = true;

#[cfg(feature = "pairings")]
mod pairings;

#[cfg(feature = "pairings")]
pub use pairings::{pairing, Bls12, Gt, MillerLoopResult};

#[cfg(feature = "pairings")]
pub use pairings::{multi_miller_loop, G2Prepared};

#[cfg(feature = "hashing")]
mod isogeny;
