// Minimal ICML test
#import "@preview/lucky-icml:0.7.0": icml2025

#show: icml2025.with(
  title: [Test Paper],
  authors: (
    (
      name: "Anonymous",
      affiliation: "Anonymous Institution",
    ),
  ),
  abstract: [
    This is a test abstract.
  ],
  bibliography: none,
  accepted: false,
)

= Introduction

This is a test document.

== Section 1

Some content here.
