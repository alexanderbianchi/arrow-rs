// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::cast::*;
use arrow_array::builder::{BinaryRunBuilder, PrimitiveRunBuilder, StringRunBuilder};
use arrow_array::downcast_primitive_array;
use arrow_array::types::RunEndIndexType;
use arrow_array::RunArray;
use arrow_schema::{ArrowError, DataType};
use std::sync::Arc;

/// Attempts to cast a `RunArray` with run-end type R to
/// `to_type` for supported types.
///
/// R is the run-end type
pub(crate) fn run_end_cast<R: RunEndIndexType>(
    array: &dyn Array,
    to_type: &DataType,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    use DataType::*;

    match to_type {
        RunEndEncoded(run_ends_type, values_type) => {
            let run_array = array
                .as_any()
                .downcast_ref::<RunArray<R>>()
                .ok_or_else(|| {
                    ArrowError::ComputeError(
                        "Internal Error: Cannot cast run array to RunArray of expected type"
                            .to_string(),
                    )
                })?;

            // Cast values array directly without expanding
            let cast_values =
                cast_with_options(run_array.values(), values_type.data_type(), cast_options)?;

            // Get run ends as a PrimitiveArray
            let run_ends_data = run_array.to_data().child_data()[0].clone();
            let run_ends = PrimitiveArray::<R>::from(run_ends_data);

            // Cast run ends if needed
            match run_ends_type.data_type() {
                dt if dt == &R::DATA_TYPE => {
                    // Same run end type, just rebuild with cast values
                    Ok(Arc::new(RunArray::<R>::try_new(&run_ends, &cast_values)?))
                }
                Int16 => {
                    let cast_run_ends = cast_run_ends::<R, Int16Type>(&run_ends, cast_options)?;
                    Ok(Arc::new(RunArray::<Int16Type>::try_new(
                        &cast_run_ends,
                        &cast_values,
                    )?))
                }
                Int32 => {
                    let cast_run_ends = cast_run_ends::<R, Int32Type>(&run_ends, cast_options)?;
                    Ok(Arc::new(RunArray::<Int32Type>::try_new(
                        &cast_run_ends,
                        &cast_values,
                    )?))
                }
                Int64 => {
                    let cast_run_ends = cast_run_ends::<R, Int64Type>(&run_ends, cast_options)?;
                    Ok(Arc::new(RunArray::<Int64Type>::try_new(
                        &cast_run_ends,
                        &cast_values,
                    )?))
                }
                dt => Err(ArrowError::CastError(format!(
                    "Unsupported run-end index type: {dt:?}"
                ))),
            }
        }
        _ => unpack_run_array::<R>(array, to_type, cast_options),
    }
}

/// Unpack a run-end encoded array into a flattened array of type to_type
pub(crate) fn unpack_run_array<R>(
    array: &dyn Array,
    to_type: &DataType,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError>
where
    R: RunEndIndexType,
{
    let run_array = array
        .as_any()
        .downcast_ref::<RunArray<R>>()
        .ok_or_else(|| {
            ArrowError::ComputeError("Internal Error: Cannot downcast to RunArray".to_string())
        })?;
    let expanded = expand_run_array(run_array)?;
    cast_with_options(&expanded, to_type, cast_options)
}

/// Attempts to encode an array into a `RunArray` with run-end
/// type R
///
/// R is the run-end type
pub(crate) fn cast_to_run_array<R: RunEndIndexType>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    use DataType::*;

    // First try to handle primitive types using the macro
    let result = downcast_primitive_array! {
        array => build_primitive_run_array::<R, _>(array),
        Boolean => build_boolean_run_array::<R>(array),
        Utf8 => build_string_run_array::<R, i32>(array),
        LargeUtf8 => build_string_run_array::<R, i64>(array),
        Binary => build_binary_run_array::<R, i32>(array),
        LargeBinary => build_binary_run_array::<R, i64>(array),
        // Handle other types generically
        _ => build_run_array_generic::<R>(array, cast_options)
    };

    result
}

/// Get the maximum run end value for a given RunEndIndexType
#[inline]
fn max_run_end_value<R: RunEndIndexType>() -> usize {
    match R::DATA_TYPE {
        DataType::Int16 => i16::MAX as usize,
        DataType::Int32 => i32::MAX as usize,
        DataType::Int64 => i64::MAX as usize,
        _ => unreachable!("Invalid run end index type"),
    }
}

/// Helper function to build a run array from a primitive array
///
/// This function handles all primitive types that implement ArrowPrimitiveType
/// by using the PrimitiveRunBuilder to automatically handle run-length encoding
fn build_primitive_run_array<R, V>(array: &PrimitiveArray<V>) -> Result<ArrayRef, ArrowError>
where
    R: RunEndIndexType,
    V: ArrowPrimitiveType,
{
    // Check if the array length could overflow the run end type
    let max_run_end = max_run_end_value::<R>();
    if array.len() > max_run_end {
        return Err(ArrowError::CastError(format!(
            "Can't cast value {} to type {}",
            array.len(),
            R::DATA_TYPE
        )));
    }

    let mut builder = PrimitiveRunBuilder::<R, V>::new();

    for i in 0..array.len() {
        match array.is_null(i) {
            true => builder.append_null(),
            false => {
                let value = array.value(i);
                builder.append_value(value);
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Build a run array from a boolean array
///
/// Boolean arrays require special handling as they don't implement ArrowPrimitiveType,
/// so we manually track runs and build the result arrays
fn build_boolean_run_array<R: RunEndIndexType>(array: &dyn Array) -> Result<ArrayRef, ArrowError> {
    let boolean_array = array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            ArrowError::ComputeError("Internal Error: Cannot downcast to BooleanArray".to_string())
        })?;

    // Unlike other primitive types, BooleanType doesn't implement ArrowPrimitiveType
    // so we can't use PrimitiveRunBuilder. We need to manually track runs.
    let mut run_ends = Vec::new();
    let mut values = Vec::new();
    let mut current_run_end = 0;
    let mut last_value: Option<Option<bool>> = None;

    for i in 0..boolean_array.len() {
        let current_value = if boolean_array.is_null(i) {
            None
        } else {
            Some(boolean_array.value(i))
        };

        if last_value != Some(current_value) {
            if i > 0 {
                run_ends.push(R::Native::from_usize(current_run_end).ok_or_else(|| {
                    ArrowError::CastError(format!(
                        "Cannot encode array into run array with {} run ends: run end value {} exceeds capacity",
                        R::DATA_TYPE,
                        current_run_end
                    ))
                })?);
                values.push(last_value.unwrap());
            }
            last_value = Some(current_value);
        }

        current_run_end = i + 1;
    }

    // Push final run
    if let Some(last) = last_value {
        run_ends.push(R::Native::from_usize(current_run_end).ok_or_else(|| {
            ArrowError::CastError(format!(
                "Cannot encode array into run array with {} run ends: run end value {} exceeds capacity",
                R::DATA_TYPE,
                current_run_end
            ))
        })?);
        values.push(last);
    }

    let run_ends = PrimitiveArray::<R>::from_iter_values(run_ends);
    let values = BooleanArray::from(values);

    Ok(Arc::new(RunArray::<R>::try_new(&run_ends, &values)?))
}

/// Build a run array from a string array
///
/// Uses the StringRunBuilder which handles UTF-8 validation and run-length encoding
fn build_string_run_array<R, O>(array: &dyn Array) -> Result<ArrayRef, ArrowError>
where
    R: RunEndIndexType,
    O: OffsetSizeTrait,
{
    // Check if the array length could overflow the run end type
    let max_run_end = max_run_end_value::<R>();
    if array.len() > max_run_end {
        return Err(ArrowError::CastError(format!(
            "Can't cast value {} to type {}",
            array.len(),
            R::DATA_TYPE
        )));
    }

    let string_array = array.as_string::<O>();
    let mut builder = StringRunBuilder::<R>::new();

    for i in 0..array.len() {
        match string_array.is_null(i) {
            true => builder.append_null(),
            false => {
                let value = string_array.value(i);
                builder.append_value(value);
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Build a run array from a binary array
///
/// Uses the BinaryRunBuilder to handle arbitrary binary data with run-length encoding
fn build_binary_run_array<R, O>(array: &dyn Array) -> Result<ArrayRef, ArrowError>
where
    R: RunEndIndexType,
    O: OffsetSizeTrait,
{
    // Check if the array length could overflow the run end type
    let max_run_end = max_run_end_value::<R>();
    if array.len() > max_run_end {
        return Err(ArrowError::CastError(format!(
            "Can't cast value {} to type {}",
            array.len(),
            R::DATA_TYPE
        )));
    }

    let binary_array = array.as_binary::<O>();
    let mut builder = BinaryRunBuilder::<R>::new();

    for i in 0..array.len() {
        match binary_array.is_null(i) {
            true => builder.append_null(),
            false => {
                let value = binary_array.value(i);
                builder.append_value(value);
            }
        }
    }

    Ok(Arc::new(builder.finish()))
}

/// Fallback for complex types where we need generic comparison
///
/// This handles types like List, Struct, Union, etc. that don't have specialized
/// run builders. It manually tracks runs by comparing consecutive elements.
fn build_run_array_generic<R: RunEndIndexType>(
    array: &dyn Array,
    cast_options: &CastOptions,
) -> Result<ArrayRef, ArrowError> {
    if array.is_empty() {
        let empty_run_ends = PrimitiveArray::<R>::from_iter_values(std::iter::empty());
        return Ok(Arc::new(RunArray::<R>::try_new(
            &empty_run_ends,
            &new_empty_array(array.data_type()),
        )?));
    }

    let mut run_ends_builder = Vec::<R::Native>::new();
    let mut values_indices = Vec::new();
    let mut last_value_index = 0;
    let mut current_end = 1;

    while current_end <= array.len() {
        if current_end == array.len() || !is_equal_at(array, current_end - 1, current_end)? {
            match R::Native::from_usize(current_end) {
                Some(run_end) => {
                    run_ends_builder.push(run_end);
                    values_indices.push(last_value_index);
                    last_value_index = current_end;
                }
                None => {
                    if cast_options.safe {
                        break;
                    } else {
                        return Err(ArrowError::CastError(format!(
                            "Cannot encode array into run array with {} run ends: run end value {} exceeds capacity",
                            R::DATA_TYPE,
                            current_end
                        )));
                    }
                }
            }
        }
        current_end += 1;
    }

    if run_ends_builder.is_empty() {
        let empty_run_ends = PrimitiveArray::<R>::from_iter_values(std::iter::empty());
        return Ok(Arc::new(RunArray::<R>::try_new(
            &empty_run_ends,
            &new_empty_array(array.data_type()),
        )?));
    }

    let run_ends = PrimitiveArray::<R>::from_iter_values(run_ends_builder);
    let indices = Int32Array::from(values_indices.iter().map(|&i| i as i32).collect::<Vec<_>>());
    let values = take(array, &indices, None)?;

    Ok(Arc::new(RunArray::<R>::try_new(&run_ends, &values)?))
}

// Helper functions

/// Expand a run-encoded array back to its logical form
///
/// This creates an array where each logical index maps to its corresponding value
/// in the run array's values array.
fn expand_run_array<R: RunEndIndexType>(run_array: &RunArray<R>) -> Result<ArrayRef, ArrowError> {
    let values = run_array.values();
    let total_len = run_array.len();
    let mut indices = Vec::with_capacity(total_len);

    for i in 0..total_len {
        indices.push(run_array.get_physical_index(i) as i32);
    }

    let indices_array = Int32Array::from(indices);
    take(values.as_ref(), &indices_array, None)
}

/// Check if two elements at the given indices are equal
///
/// This handles null values and uses type-specific comparison for performance
/// on primitive types, falling back to slice comparison for complex types.
fn is_equal_at(array: &dyn Array, left_idx: usize, right_idx: usize) -> Result<bool, ArrowError> {
    if left_idx >= array.len() || right_idx >= array.len() {
        return Ok(false);
    }

    // Handle nulls
    match (array.is_null(left_idx), array.is_null(right_idx)) {
        (true, true) => Ok(true),
        (false, false) => {
            // Use direct value comparison for primitive types
            use DataType::*;
            match array.data_type() {
                Boolean => {
                    let arr = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    Ok(arr.value(left_idx) == arr.value(right_idx))
                }
                Int8 => downcast_primitive_compare::<Int8Type>(array, left_idx, right_idx),
                Int16 => downcast_primitive_compare::<Int16Type>(array, left_idx, right_idx),
                Int32 => downcast_primitive_compare::<Int32Type>(array, left_idx, right_idx),
                Int64 => downcast_primitive_compare::<Int64Type>(array, left_idx, right_idx),
                UInt8 => downcast_primitive_compare::<UInt8Type>(array, left_idx, right_idx),
                UInt16 => downcast_primitive_compare::<UInt16Type>(array, left_idx, right_idx),
                UInt32 => downcast_primitive_compare::<UInt32Type>(array, left_idx, right_idx),
                UInt64 => downcast_primitive_compare::<UInt64Type>(array, left_idx, right_idx),
                Float16 => downcast_primitive_compare::<Float16Type>(array, left_idx, right_idx),
                Float32 => downcast_primitive_compare::<Float32Type>(array, left_idx, right_idx),
                Float64 => downcast_primitive_compare::<Float64Type>(array, left_idx, right_idx),
                Utf8 => {
                    let arr = array.as_any().downcast_ref::<StringArray>().unwrap();
                    Ok(arr.value(left_idx) == arr.value(right_idx))
                }
                LargeUtf8 => {
                    let arr = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
                    Ok(arr.value(left_idx) == arr.value(right_idx))
                }
                Binary => {
                    let arr = array.as_any().downcast_ref::<BinaryArray>().unwrap();
                    Ok(arr.value(left_idx) == arr.value(right_idx))
                }
                LargeBinary => {
                    let arr = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
                    Ok(arr.value(left_idx) == arr.value(right_idx))
                }
                _ => {
                    // For complex types (List, Struct, Union, etc.), use slice comparison
                    let left = array.slice(left_idx, 1);
                    let right = array.slice(right_idx, 1);
                    Ok(left.as_ref() == right.as_ref())
                }
            }
        }
        _ => Ok(false),
    }
}

/// Helper to downcast and compare primitive array values
#[inline]
fn downcast_primitive_compare<T: ArrowPrimitiveType>(
    array: &dyn Array,
    left_idx: usize,
    right_idx: usize,
) -> Result<bool, ArrowError> {
    let arr = array.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();
    Ok(arr.value(left_idx) == arr.value(right_idx))
}

/// Helper to cast run ends from one type to another
fn cast_run_ends<R, T>(
    run_ends: &PrimitiveArray<R>,
    cast_options: &CastOptions,
) -> Result<PrimitiveArray<T>, ArrowError>
where
    R: RunEndIndexType,
    T: RunEndIndexType,
{
    let cast_run_ends = cast_with_options(run_ends, &T::DATA_TYPE, cast_options)?;
    cast_run_ends
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .cloned()
        .ok_or_else(|| {
            ArrowError::ComputeError("Internal Error: Cannot downcast run ends".to_string())
        })
}
