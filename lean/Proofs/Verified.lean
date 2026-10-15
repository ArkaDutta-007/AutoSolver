-- Verified lemmas accumulated by autoresearch.
-- Every entry compiled cleanly when added. Import with:
--   import Proofs.Verified
import Mathlib
set_option maxHeartbeats 400000


-- For every natural number n, 2 divides n*(n+1).

theorem two_divides_product_of_consecutive_naturals (n : ℕ) : 2 ∣ n * (n + 1) := by exact?

-- For every natural number n, 2 divides n*(n+1)

theorem gcd_divides {m n : Nat} :
  gcd m n ∣ m ∧ gcd m n ∣ n := by constructor <;> exact?

-- For positive integers m, n, gcd(m,n) divides both m and n.

theorem gcd_divides_both {m n : ℕ} (h : m > 0 ∧ n > 0) : Nat.gcd m n ∣ m ∧ Nat.gcd m n ∣ n := by exact?

-- For positive integers m, n, gcd(m,n) divides both m and n.

-- For all natural numbers a and b, gcd(min(a,b), max(a,b) - min(a,b)) = gcd(a,b). (The Euclid step: replacing the pair of p-exponents (a,b) by (min(a,b)
theorem gcd_min_diff : ∀ a b : ℕ, Nat.gcd (min a b) (max a b - min a b) = Nat.gcd a b := by
  intro a b
  have h₁ : Nat.gcd (min a b) (max a b - min a b) = Nat.gcd a b := by
    have h₂ : a ≤ b ∨ b ≤ a := by omega
    cases h₂ with
    | inl h₂ =>
      -- Case: a ≤ b
      have h₃ : min a b = a := by
        rw [min_eq_left h₂]
      have h₄ : max a b = b := by
        rw [max_eq_right h₂]
      rw [h₃, h₄]
      have h₅ : b - a = b - a := rfl
      have h₆ : Nat.gcd a (b - a) = Nat.gcd a b := by
        have h₇ : Nat.gcd a (b - a) = Nat.gcd a b := by
          rw [← Nat.sub_add_cancel h₂]
          <;> simp [Nat.gcd_comm]
          <;>
          simp_all [Nat.gcd_comm, Nat.gcd_assoc, Nat.gcd_add_mul_right_right]
          <;>
          ring_nf at *
          <;>
          simp_all [Nat.gcd_comm, Nat.gcd_assoc, Nat.gcd_add_mul_right_right]
        exact h₇
      rw [h₆]
    | inr h₂ =>
      -- Case: b ≤ a
      have h₃ : min a b = b := by
        rw [min_eq_right h₂]
      have h₄ : max a b = a := by
        rw [max_eq_left h₂]
      rw [h₃, h₄]
      have h₅ : a - b = a - b := rfl
      have h₆ : Nat.gcd b (a - b) = Nat.gcd a b := by
        have h₇ : Nat.gcd b (a - b) = Nat.gcd a b := by
          have h₈ : Nat.gcd b (a - b) = Nat.gcd (a - b) b := by
            rw [Nat.gcd_comm]
          rw [h₈]
          have h₉ : Nat.gcd (a - b) b = Nat.gcd a b := by
            have h₁₀ : a - b + b = a := by
              have h₁₁ : b ≤ a := h₂
              have h₁₂ : a - b + b = a := by
                rw [Nat.sub_add_cancel h₁₁]
              exact h₁₂
            have h₁₁ : Nat.gcd (a - b) b = Nat.gcd a b := by
              rw [← h₁₀]
              <;> simp [Nat.gcd_comm]
              <;>
              simp_all [Nat.gcd_comm, Nat.gcd_assoc, Nat.gcd_add_mul_right_right]
              <;>
              ring_nf at *
              <;>
              simp_all [Nat.gcd_comm, Nat.gcd_assoc, Nat.gcd_add_mul_right_right]
            exact h₁₁
          rw [h₉]
          <;> simp [Nat.gcd_comm]
        exact h₇
      rw [h₆]
  exact h₁

-- Simplified Shan-Yu defense lemma for IMO26_P4. Let a,b,c,x,theta be real numbers with a+b+c = 180. Suppose none of a, b, c is an integer multiple of t
open scoped Matrix BigOperators Topology NNReal ENNReal Pointwise

theorem shanyu_defense_lemma_v2 (a b c x θ : ℝ)
    (hs : a + b + c = 180)
    (haD : ¬ ∃ k : ℤ, a = ↑k * θ)
    (hbD : ¬ ∃ k : ℤ, b = ↑k * θ)
    (hcD : ¬ ∃ k : ℤ, c = ↑k * θ)
    (hsD : ¬ ∃ k : ℤ, 180 = ↑k * θ) :
    ((∃ k : ℤ, x = ↑k * θ) ∨ (∃ k : ℤ, a + c - x = ↑k * θ)) →
      (¬ ∃ m : ℤ, a - x = ↑m * θ) ∧ (¬ ∃ m : ℤ, b + x = ↑m * θ) := by
  intro h
  have h₁ : ¬ ∃ m : ℤ, a - x = ↑m * θ := by
    intro h₂
    -- Assume there exists an integer m such that a - x = m * θ
    cases' h with h₃ h₃
    · -- Case 1: There exists an integer k such that x = k * θ
      obtain ⟨k, hk⟩ := h₃
      obtain ⟨m, hm⟩ := h₂
      -- Derive a = (m + k) * θ, contradicting haD
      have h₄ : a = (↑m + ↑k) * θ := by
        have h₅ : (a : ℝ) - x = (m : ℝ) * θ := by exact_mod_cast hm
        have h₆ : (x : ℝ) = (k : ℝ) * θ := by exact_mod_cast hk
        have h₇ : (a : ℝ) - (k : ℝ) * θ = (m : ℝ) * θ := by linarith
        have h₈ : (a : ℝ) = (m : ℝ) * θ + (k : ℝ) * θ := by linarith
        have h₉ : (a : ℝ) = (↑m + ↑k : ℝ) * θ := by
          ring_nf at h₈ ⊢
          <;> linarith
        exact_mod_cast h₉
      have h₅ : ∃ (k : ℤ), a = (k : ℝ) * θ := ⟨m + k, by exact_mod_cast h₄⟩
      exact haD h₅
    · -- Case 2: There exists an integer k such that a + c - x = k * θ
      obtain ⟨k, hk⟩ := h₃
      obtain ⟨m, hm⟩ := h₂
      -- Derive c = (k - m) * θ, contradicting hcD
      have h₄ : c = (↑k - ↑m) * θ := by
        have h₅ : (a : ℝ) - x = (m : ℝ) * θ := by exact_mod_cast hm
        have h₆ : (a : ℝ) + c - x = (k : ℝ) * θ := by exact_mod_cast hk
        have h₇ : (c : ℝ) = (k : ℝ) * θ - (m : ℝ) * θ := by linarith
        have h₈ : (c : ℝ) = (↑k - ↑m : ℝ) * θ := by
          ring_nf at h₇ ⊢
          <;> linarith
        exact_mod_cast h₈
      have h₅ : ∃ (k : ℤ), c = (k : ℝ) * θ := ⟨k - m, by exact_mod_cast h₄⟩
      exact hcD h₅
  
  have h₂ : ¬ ∃ m : ℤ, b + x = ↑m * θ := by
    intro h₃
    -- Assume there exists an integer m such that b + x = m * θ
    cases' h with h₄ h₄
    · -- Case 1: There exists an integer k such that x = k * θ
      obtain ⟨k, hk⟩ := h₄
      obtain ⟨m, hm⟩ := h₃
      -- Derive b = (m - k) * θ, contradicting hbD
      have h₅ : b = (↑m - ↑k) * θ := by
        have h₆ : (b : ℝ) + x = (m : ℝ) * θ := by exact_mod_cast hm
        have h₇ : (x : ℝ) = (k : ℝ) * θ := by exact_mod_cast hk
        have h₈ : (b : ℝ) + (k : ℝ) * θ = (m : ℝ) * θ := by linarith
        have h₉ : (b : ℝ) = (m : ℝ) * θ - (k : ℝ) * θ := by linarith
        have h₁₀ : (b : ℝ) = (↑m - ↑k : ℝ) * θ := by
          ring_nf at h₉ ⊢
          <;> linarith
        exact_mod_cast h₁₀
      have h₆ : ∃ (k : ℤ), b = (k : ℝ) * θ := ⟨m - k, by exact_mod_cast h₅⟩
      exact hbD h₆
    · -- Case 2: There exists an integer k such that a + c - x = k * θ
      obtain ⟨k, hk⟩ := h₄
      obtain ⟨m, hm⟩ := h₃
      -- Derive 180 = (k + m) * θ, contradicting hsD
      have h₅ : (180 : ℝ) = (↑k + ↑m) * θ := by
        have h₆ : (a : ℝ) + c - x = (k : ℝ) * θ := by exact_mod_cast hk
        have h₇ : (b : ℝ) + x = (m : ℝ) * θ := by exact_mod_cast hm
        have h₈ : (a : ℝ) + b + c = 180 := by
          norm_num at hs ⊢
          <;> linarith
        have h₉ : (a : ℝ) + c - x + (b : ℝ) + x = (k : ℝ) * θ + (m : ℝ) * θ := by linarith
        have h₁₀ : (a : ℝ) + b + c = (↑k + ↑m : ℝ) * θ := by linarith
        have h₁₁ : (180 : ℝ) = (↑k + ↑m : ℝ) * θ := by
          linarith
        exact_mod_cast h₁₁
      have h₆ : ∃ (k : ℤ), (180 : ℝ) = (k : ℝ) * θ := ⟨k + m, by exact_mod_cast h₅⟩
      exact hsD (by
        obtain ⟨k, hk⟩ := h₆
        refine' ⟨k, _⟩
        <;> simp_all [mul_comm]
        <;> ring_nf at *
        <;> norm_num at *
        <;> linarith)
  
  exact ⟨h₁, h₂⟩

-- For all nonnegative integers a and b, gcd(min(a,b), max(a,b) - min(a,b)) = gcd(a,b). Since max-min = |a-b| in the naturals, this is exactly the Euclid
theorem gcd_min_diff_2 (a b : ℕ) : Nat.gcd (min a b) (max a b - min a b) = Nat.gcd a b := by
  have h₁ : Nat.gcd (min a b) (max a b - min a b) = Nat.gcd a b := by
    have h₂ : a ≤ b ∨ b ≤ a := by omega
    cases h₂ with
    | inl h₂ =>
      -- Case: a ≤ b
      have h₃ : min a b = a := by
        rw [min_eq_left h₂]
      have h₄ : max a b = b := by
        rw [max_eq_right h₂]
      rw [h₃, h₄]
      have h₅ : b - a = b - a := rfl
      have h₆ : Nat.gcd a (b - a) = Nat.gcd a b := by
        have h₇ : Nat.gcd a (b - a) = Nat.gcd a b := by
          rw [← Nat.sub_add_cancel h₂]
          <;> simp [Nat.gcd_comm]
          <;> rw [Nat.gcd_comm]
          <;> simp [Nat.gcd_comm]
        exact h₇
      rw [h₆]
      <;> simp [Nat.gcd_comm]
    | inr h₂ =>
      -- Case: b ≤ a
      have h₃ : min a b = b := by
        rw [min_eq_right h₂]
      have h₄ : max a b = a := by
        rw [max_eq_left h₂]
      rw [h₃, h₄]
      have h₅ : a - b = a - b := rfl
      have h₆ : Nat.gcd b (a - b) = Nat.gcd a b := by
        have h₇ : Nat.gcd b (a - b) = Nat.gcd a b := by
          have h₈ : Nat.gcd b (a - b) = Nat.gcd (a - b) b := by
            rw [Nat.gcd_comm]
          rw [h₈]
          have h₉ : Nat.gcd (a - b) b = Nat.gcd a b := by
            have h₁₀ : a - b + b = a := by
              have h₁₁ : b ≤ a := h₂
              have h₁₂ : a - b + b = a := by
                rw [Nat.sub_add_cancel h₁₁]
              exact h₁₂
            have h₁₁ : Nat.gcd (a - b) b = Nat.gcd a b := by
              rw [← h₁₀]
              <;> simp [Nat.gcd_comm]
              <;> rw [Nat.gcd_comm]
              <;> simp [Nat.gcd_comm]
            exact h₁₁
          rw [h₉]
          <;> simp [Nat.gcd_comm]
        exact h₇
      rw [h₆]
      <;> simp [Nat.gcd_comm]
  exact h₁

-- For all positive integers m, n and every natural number p: the exponent of p in lcm(m,n)/gcd(m,n) equals max(v_p(m), v_p(n)) - min(v_p(m), v_p(n)), wh
theorem vp_lcm_div_gcd_nat {p m n : ℕ} (hm : 0 < m) (hn : 0 < n) :
    ((m.lcm n) / (m.gcd n)).factorization p =
      max (m.factorization p) (n.factorization p) - min (m.factorization p) (n.factorization p) := by
  have h₁ : m.lcm n / m.gcd n = m.lcm n / m.gcd n := rfl
  have h₂ : ((m.lcm n) / (m.gcd n)).factorization p = (m.lcm n).factorization p - (m.gcd n).factorization p := by
    have h₃ : m.gcd n ∣ m.lcm n := by
      have h₄ : m.gcd n ∣ m := Nat.gcd_dvd_left m n
      have h₅ : m ∣ m.lcm n := Nat.dvd_lcm_left m n
      exact dvd_trans h₄ h₅
    have h₆ : (m.lcm n / m.gcd n).factorization p = (m.lcm n).factorization p - (m.gcd n).factorization p := by
      rw [show m.lcm n / m.gcd n = m.lcm n / m.gcd n by rfl]
      -- Use the property of factorization for division
      rw [Nat.factorization_div]
      <;>
      (try norm_num) <;>
      (try simp_all [Nat.Prime.ne_zero]) <;>
      (try omega) <;>
      (try
        {
          aesop
        }) <;>
      (try
        {
          have h₇ : m.gcd n ∣ m.lcm n := h₃
          simp_all [Nat.Prime.ne_zero]
          <;>
          omega
        })
      <;>
      (try
        {
          have h₇ : m.gcd n ∣ m.lcm n := h₃
          simp_all [Nat.Prime.ne_zero]
          <;>
          omega
        })
    exact h₆
  have h₃ : (m.lcm n).factorization p = max (m.factorization p) (n.factorization p) := by
    rw [Nat.factorization_lcm]
    <;>
    simp [max_eq_left, max_eq_right, le_of_lt]
    <;>
    (try omega)
    <;>
    (try
      {
        aesop
      })
  have h₄ : (m.gcd n).factorization p = min (m.factorization p) (n.factorization p) := by
    rw [Nat.factorization_gcd]
    <;>
    simp [min_eq_left, min_eq_right, le_of_lt]
    <;>
    (try omega)
    <;>
    (try
      {
        aesop
      })
  rw [h₂, h₃, h₄]
  <;>
  simp [Nat.sub_eq_zero_of_le]
  <;>
  (try omega)
  <;>
  (try
    {
      aesop
    })
