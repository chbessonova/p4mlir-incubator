// SPDX-FileCopyrightText: 2026 The P4 Language Consortium
//
// SPDX-License-Identifier: Apache-2.0

// RUN: p4mlir-opt %s --lower-p4hir-to-llvm -split-input-file \
// RUN:   | FileCheck %s --implicit-check-not=llvm.alloca --implicit-check-not=llvm.getelementptr \
// RUN:       --implicit-check-not=llvm.load --implicit-check-not=llvm.store

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

// Reads at a dynamic index are not lowered, even from an empty array. Neither are
// reads at a constant position beyond the 32-bit range of `llvm.extractvalue`,
// which must not be truncated.

// CHECK-LABEL: @get(
// CHECK:         p4hir.array_get %{{.*}}[%{{.*}}] : !{{.*}}, !b32i
// CHECK:         p4hir.array_get %{{.*}}[%{{.*}}] : !{{.*}}, !b32i
// CHECK:         p4hir.array_get %{{.*}}[%{{.*}}] : !{{.*}}, !b64i

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!b64i = !p4hir.bit<64>
!arr = !p4hir.array<2 x !b8i>
!empty = !p4hir.array<0 x !b8i>
!large = !p4hir.array<4294967297 x !b8i>

module {
  p4hir.func @get(%arr : !arr, %empty : !empty, %large : !large, %i : !b32i) -> !b8i {
    %dyn = p4hir.array_get %arr[%i] : !arr, !b32i
    %dyn_empty = p4hir.array_get %empty[%i] : !empty, !b32i
    %pos = p4hir.const #p4hir.int<4294967296> : !b64i
    %r = p4hir.array_get %large[%pos] : !large, !b64i
    p4hir.return %r : !b8i
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
// not lowered.

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

// References into headers stay in P4HIR.

// CHECK-LABEL: @refs(
// CHECK:         p4hir.struct_field_ref %{{.*}}["h"] : <!with_header>

!b8i = !p4hir.bit<8>
!validity_bit = !p4hir.validity.bit
!h = !p4hir.header<"h", a: !b8i, __valid: !validity_bit>
!with_header = !p4hir.struct<"with_header", x: !b8i, h: !h>

module {
  p4hir.func @refs(%s : !p4hir.ref<!with_header>) {
    %h = p4hir.struct_field_ref %s["h"] : <!with_header>
    p4hir.return
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

// A call cannot carry the bounds check. The whole reference chain stays in P4HIR.

// CHECK-LABEL: @escape(
// CHECK:         p4hir.array_element_ref
// CHECK:         p4hir.call @sink
// CHECK:         p4hir.read
// CHECK-LABEL: @projected_escape(
// CHECK:         p4hir.array_element_ref
// CHECK:         p4hir.struct_field_ref
// CHECK:         p4hir.call @sink
// CHECK:         p4hir.read
// CHECK-LABEL: @oob_escape(
// CHECK:         p4hir.array_element_ref
// CHECK:         p4hir.call @sink

!b8i = !p4hir.bit<8>
!b32i = !p4hir.bit<32>
!s = !p4hir.struct<"s", x: !b8i>
!arr = !p4hir.array<4 x !b8i>
!structs = !p4hir.array<2 x !s>

module {
  p4hir.func @escape(%a : !p4hir.ref<!arr>, %i : !b32i) -> !b8i {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!arr>, !b32i
    p4hir.call @sink (%e) : (!p4hir.ref<!b8i>) -> ()
    %r = p4hir.read %e : <!b8i>
    p4hir.return %r : !b8i
  }

  p4hir.func @projected_escape(%a : !p4hir.ref<!structs>, %i : !b32i) -> !s {
    %e = p4hir.array_element_ref %a[%i] : !p4hir.ref<!structs>, !b32i
    %f = p4hir.struct_field_ref %e["x"] : <!s>
    p4hir.call @sink (%f) : (!p4hir.ref<!b8i>) -> ()
    %r = p4hir.read %e : <!s>
    p4hir.return %r : !s
  }

  p4hir.func @oob_escape(%a : !p4hir.ref<!arr>) {
    %c4 = p4hir.const #p4hir.int<4> : !b8i
    %e = p4hir.array_element_ref %a[%c4] : !p4hir.ref<!arr>, !b8i
    p4hir.call @sink (%e) : (!p4hir.ref<!b8i>) -> ()
    p4hir.return
  }

  p4hir.func @sink(%r : !p4hir.ref<!b8i>) {
    p4hir.return
  }
}
