import Mathlib

theorem two_dvd_mul_succ (n : ℕ) : 2 ∣ n * (n + 1) :=
  (Nat.even_mul_succ_self n).two_dvd
