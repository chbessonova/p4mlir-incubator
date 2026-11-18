// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s --check-prefix=NO-ACCESS
// NO-ACCESS-NOT: llvm.alloca
// NO-ACCESS-NOT: llvm.getelementptr
// NO-ACCESS-NOT: llvm.load
// NO-ACCESS-NOT: llvm.store

// Calls cannot carry the bounds predicate. Leave the entire reference chain,
// including its direct reads, unconverted.
// CHECK-LABEL: @escape(
// CHECK: p4hir.array_element_ref
// CHECK: p4hir.call @sink
// CHECK: p4hir.read
!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!arr = !p4hir.array<4 x !b8i>
module {
  p4hir.func @escape(%a: !p4hir.ref<!arr>, %i: !b32i) -> !b8i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    p4hir.call @sink (%e) : (!p4hir.ref<!b8i>) -> ()
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
  p4hir.func @sink(%r: !p4hir.ref<!b8i>) {
    p4hir.return
  }
}

// -----

// Structured regions must be flattened before guarded access lowering.
// CHECK-LABEL: @scope(
// CHECK: p4hir.array_element_ref
// CHECK: p4hir.scope {
// CHECK-NEXT: p4hir.assign
// CHECK-LABEL: @scope_read(
// CHECK: p4hir.array_element_ref
// CHECK: p4hir.scope {
// CHECK-NEXT: {{.*}} = p4hir.read
!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!arr = !p4hir.array<4 x !b8i>
module {
  p4hir.func @scope(%a: !p4hir.ref<!arr>, %i: !b32i, %v: !b8i) {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    p4hir.scope {
      p4hir.assign %v, %e : <!b8i>
    }
    p4hir.return
  }
  p4hir.func @scope_read(%a: !p4hir.ref<!arr>, %i: !b32i) {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    p4hir.scope {
      %v = p4hir.read %e : <!b8i>
    }
    p4hir.return
  }
}

// -----

// An escaping field also prevents lowering its ancestor and sibling accesses.
// CHECK-LABEL: @projected_escape(
// CHECK: p4hir.array_element_ref
// CHECK: p4hir.struct_field_ref
// CHECK: p4hir.call @sink
// CHECK: p4hir.read
!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!s = !p4hir.struct<"s", x: !b8i>
!arr = !p4hir.array<2 x !s>
module {
  p4hir.func @projected_escape(%a: !p4hir.ref<!arr>, %i: !b32i) -> !s {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    %f = p4hir.struct_field_ref %e["x"] : <!s>
    p4hir.call @sink (%f) : (!p4hir.ref<!b8i>) -> ()
    %r = p4hir.read %e : <!s>
    p4hir.return %r : !s
  }
  p4hir.func @sink(%r: !p4hir.ref<!b8i>) { p4hir.return }
}

// -----

// Constant OOB references cannot be passed to calls either.
// CHECK-LABEL: @oob_escape(
// CHECK: p4hir.array_element_ref
// CHECK: p4hir.call @sink
!b8i = !p4hir.bit<8>
!arr = !p4hir.array<2 x !b8i>
module {
  p4hir.func @oob_escape(%a: !p4hir.ref<!arr>) {
    %i = p4hir.const #p4hir.int<2> : !b8i
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b8i
    p4hir.call @sink (%e) : (!p4hir.ref<!b8i>) -> ()
    p4hir.return
  }
  p4hir.func @sink(%r: !p4hir.ref<!b8i>) { p4hir.return }
}

// -----

// Expand slice accesses into reads/writes before lowering guarded references.
// CHECK-LABEL: @slices(
// CHECK: p4hir.array_element_ref
// CHECK: p4hir.read_slice
// CHECK: p4hir.assign_slice
// CHECK: p4hir.read
!b4i = !p4hir.bit<4>
!b8i = !p4hir.bit<8>
!arr = !p4hir.array<2 x !b8i>
module {
  p4hir.func @slices(%a: !p4hir.ref<!arr>, %i: !b8i) -> !b8i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b8i
    %v = p4hir.read_slice %e[3 : 0] : <!b8i> -> !b4i
    p4hir.assign_slice %v, %e[7 : 4] : !b4i -> <!b8i>
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
}

// -----

// Infinite-precision runtime indices have no LLVM integer representation.
// CHECK-LABEL: @infint_index(
// CHECK: p4hir.array_element_ref
// CHECK: p4hir.read
!b8i = !p4hir.bit<8>
!arr = !p4hir.array<2 x !b8i>
module {
  p4hir.func @infint_index(%a: !p4hir.ref<!arr>, %i: !p4hir.infint) -> !b8i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !p4hir.infint
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }
}
