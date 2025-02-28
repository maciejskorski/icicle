// Copyright 2023 Ingonyama
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated by Ingonyama DO NOT EDIT

package bls12381

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestG2Eqg2(t *testing.T) {
	var point G2Point

	point.Random()

	assert.True(t, point.Eq(&point))
}

func TestG2FromProjectiveToAffine(t *testing.T) {
	var projective G2Point
	projective.Random()

	var affine G2PointAffine
	affine.FromProjective(&projective)

	var projective2 G2Point
	projective2.FromAffine(&affine)

	assert.True(t, projective.IsOnCurve())
	assert.True(t, projective2.IsOnCurve())
	assert.True(t, projective.Eq(&projective2))
}

func TestG2Eqg2NotEqual(t *testing.T) {
	var point G2Point
	point.Random()

	var point2 G2Point
	point2.Random()

	assert.False(t, point.Eq(&point2))
}

func TestG2ToBytes(t *testing.T) {
	element := G2Element{0x6546098ea84b6298, 0x4a384533d1f68aca, 0xaa0666972d771336, 0x1569e4a34321993}
	bytes := element.ToBytesLe()

	assert.Equal(t, bytes, []byte{0x98, 0x62, 0x4b, 0xa8, 0x8e, 0x9, 0x46, 0x65, 0xca, 0x8a, 0xf6, 0xd1, 0x33, 0x45, 0x38, 0x4a, 0x36, 0x13, 0x77, 0x2d, 0x97, 0x66, 0x6, 0xaa, 0x93, 0x19, 0x32, 0x34, 0x4a, 0x9e, 0x56, 0x1})
}

func TestG2ShouldConvertToProjective(t *testing.T) {
	fmt.Print() // this prevents the test from hanging. TODO: figure out why
	var pointProjective G2Point
	pointProjective.Random()

	var pointAffine G2PointAffine
	pointAffine.FromProjective(&pointProjective)

	proj := pointAffine.ToProjective()

	assert.True(t, proj.IsOnCurve())
	assert.True(t, pointProjective.Eq(&proj))
}
