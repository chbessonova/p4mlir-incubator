// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file | FileCheck %s

// Headers, header unions and header stacks are not lowered yet, and neither are
// structs and arrays containing them or other types without an LLVM counterpart:
// variables of these types are left unconverted.

!b8i = !p4hir.bit<8>
!infint = !p4hir.infint
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!u = !p4hir.header_union<"u", h: !h>
!hs = !p4hir.header_stack<2x!h>
!with_header = !p4hir.struct<"with_header", x: !b8i, h: !h>
!with_infint = !p4hir.struct<"with_infint", x: !b8i, i: !infint>

// CHECK-LABEL: @types(
// CHECK-NEXT:    p4hir.variable ["header", init] : <!h>
// CHECK-NEXT:    p4hir.variable ["header_union", init] : <!u>
// CHECK-NEXT:    p4hir.variable ["header_stack", init] : <!{{.*}}>
// CHECK-NEXT:    p4hir.variable ["struct_with_header", init] : <!with_header>
// CHECK-NEXT:    p4hir.variable ["struct_with_infint", init] : <!with_infint>
// CHECK-NEXT:    p4hir.variable ["array_of_infint", init] : <!{{.*}}>
// CHECK-NEXT:    p4hir.variable ["array_of_structs_with_header", init] : <!{{.*}}>

module {
  p4hir.func @types() {
    %header = p4hir.variable ["header", init] : <!h>
    %header_union = p4hir.variable ["header_union", init] : <!u>
    %header_stack = p4hir.variable ["header_stack", init] : <!hs>
    %struct_with_header = p4hir.variable ["struct_with_header", init] : <!with_header>
    %struct_with_infint = p4hir.variable ["struct_with_infint", init] : <!with_infint>
    %array_of_infint = p4hir.variable ["array_of_infint", init] : <!p4hir.array<2 x !infint>>
    %array_of_structs_with_header = p4hir.variable ["array_of_structs_with_header", init] : <!p4hir.array<2 x !with_header>>
    p4hir.return
  }
}

// -----

// Dynamic array value indexing remains unsupported.

// CHECK-LABEL: @get(
// CHECK:         p4hir.array_get %{{.*}}[%{{.*}}] : !{{.*}}, !b32i

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!arr = !p4hir.array<2 x !b8i>

module {
  p4hir.func @get(%arr : !arr, %i : !b32i) -> !b8i {
    %dyn = p4hir.array_get %arr[%i] : !arr, !b32i
    p4hir.return %dyn : !b8i
  }
}

// -----

// Building and reading headers, or structs containing them, is left unconverted.

// CHECK-LABEL: @header(
// CHECK:         p4hir.struct_extract %{{.*}}["h"] : !with_header
// CHECK:         p4hir.struct_extract %{{.*}}["a"] : !h
// CHECK:         p4hir.struct (%{{.*}}, %{{.*}}) : !h

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!with_header = !p4hir.struct<"with_header", x: !b8i, h: !h>

module {
  p4hir.func @header(%s : !with_header, %v : !validity_bit) -> !b8i {
    %h = p4hir.struct_extract %s["h"] : !with_header
    %a = p4hir.struct_extract %h["a"] : !h
    %n = p4hir.struct (%a, %v) : !h
    p4hir.return %a : !b8i
  }
}

// -----

// Constructor parameters are only known at instantiation, and header constants are
// not lowered yet.

// CHECK-LABEL: module
// CHECK:         p4hir.const ["cfg"] #{{.*}}
// CHECK:         p4hir.const #p4hir.aggregate<[#{{.*}}, #valid]> : !h

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!s = !p4hir.struct<"s", x: !b8i>
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!C = !p4hir.control<"C", ()>
#int1_b8i = #p4hir.int<1> : !b8i
#valid = #p4hir<validity.bit valid> : !validity_bit
#C_cfg = #p4hir.ctor_param<@C, "cfg"> : !s

module {
  p4hir.control @C()(cfg: !s) {
    %cfg = p4hir.const ["cfg"] #C_cfg
    p4hir.control_apply {
    }
  }
  p4hir.func @header() {
    %h = p4hir.const #p4hir.aggregate<[#int1_b8i, #valid]> : !h
    p4hir.return
  }
}

// -----

// References into headers stay in P4HIR. So do potentially out-of-bounds
// array references and their accesses outside a flattened function CFG.

// CHECK-LABEL: @refs(
// CHECK:         p4hir.struct_field_ref %{{.*}}["h"] : <!with_header>
// CHECK-LABEL: p4hir.control @C(
// CHECK:         %[[E:.*]] = p4hir.array_element_ref %{{.*}}[%{{.*}}] : !p4hir.ref<!{{.*}}>, !b32i
// CHECK-NEXT:    p4hir.read %[[E]] : <!b8i>

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!with_header = !p4hir.struct<"with_header", x: !b8i, h: !h>
!arr = !p4hir.array<2 x !b8i>
#inout = #p4hir<dir inout>
#in = #p4hir<dir in>

module {
  p4hir.func @refs(%s : !p4hir.ref<!with_header>) {
    %h = p4hir.struct_field_ref %s["h"] : <!with_header>
    p4hir.return
  }

  p4hir.control @C(%arr: !p4hir.ref<!arr> {p4hir.dir = #inout, p4hir.param_name = "arr"},
                   %i: !b32i {p4hir.dir = #in, p4hir.param_name = "i"})() {
    p4hir.control_apply {
      %e = p4hir.array_element_ref %arr[%i] : !p4hir.ref<!arr>, !b32i
      %r = p4hir.read %e : <!b8i>
    }
  }
}

// -----

// Equality of headers, and so of structs containing them, depends on validity and
// is left unconverted.

// CHECK-LABEL: @eq(
// CHECK:         p4hir.cmp(eq, %{{.*}} : !with_header, %{{.*}} : !with_header)

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!with_header = !p4hir.struct<"with_header", x: !b8i, h: !h>

module {
  p4hir.func @eq(%a : !with_header, %b : !with_header) -> !p4hir.bool {
    %eq = p4hir.cmp(eq, %a : !with_header, %b : !with_header)
    p4hir.return %eq : !p4hir.bool
  }
}
// -----

// The indexable interface cannot represent a field count of 2^32.
// CHECK-LABEL: @oversized_equality(
// CHECK-NOT: llvm.extractvalue
// CHECK: p4hir.cmp(eq,
!large = !p4hir.array<4294967296 x !p4hir.bit<0>>
module {
  p4hir.func @oversized_equality(%a: !large, %b: !large) -> !p4hir.bool {
    %eq = p4hir.cmp(eq, %a : !large, %b : !large)
    p4hir.return %eq : !p4hir.bool
  }
}
// -----

// A dynamic index remains unsupported even for an empty array. A valid position
// beyond LLVM's 32-bit extractvalue encoding must not be silently truncated.
// CHECK-LABEL: @unsupported(
// CHECK: p4hir.array_get
// CHECK: p4hir.array_get
!b8i = !p4hir.bit<8>
!b64i = !p4hir.bit<64>
!large = !p4hir.array<4294967297 x !b8i>
!empty = !p4hir.array<0 x !b8i>
module {
  p4hir.func @unsupported(%a: !large, %empty: !empty, %i: !b64i) -> !b8i {
    %dyn = p4hir.array_get %empty[%i] : !empty, !b64i
    %pos = p4hir.const #p4hir.int<4294967296> : !b64i
    %r = p4hir.array_get %a[%pos] : !large, !b64i
    p4hir.return %r : !b8i
  }
}
