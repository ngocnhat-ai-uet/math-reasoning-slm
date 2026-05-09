import unittest

import math_answer_matcher as matcher


def result_for(label: str, extracted: str) -> matcher.MatchResult:
    return matcher.match_answer(label, rf"\boxed{{{extracted}}}")


class MathAnswerMatcherRegressionTest(unittest.TestCase):
    def assert_matches(self, label: str, extracted: str) -> None:
        result = result_for(label, extracted)
        self.assertTrue(
            result.matched,
            msg=f"{label!r} should match {extracted!r}; got {result.reason}",
        )

    def assert_no_match(self, label: str, extracted: str) -> None:
        result = result_for(label, extracted)
        self.assertFalse(
            result.matched,
            msg=f"{label!r} should not match {extracted!r}; got {result.reason}",
        )

    def test_format_equivalent_latex(self) -> None:
        self.assert_matches(r"\frac{26}{15}", r"\dfrac{26}{15}")
        self.assert_matches(r"\frac{10^{-2.5}}{\gamma_{\text{HCl}}}", r"\dfrac{10^{-2.5}}{\gamma_{HCl}}")
        self.assert_matches(r"\text{odd}", "odd")
        self.assert_matches("Panthera", r"\text{Panthera}")
        self.assert_matches(r"(6, -17)", r"\left( 6, -17 \right)")
        self.assert_matches(r"[-3, -\frac{4}{3})", r"[-3, -\dfrac{4}{3})")
        self.assert_matches(
            r"\begin{pmatrix} -3 & 0 \\ 0 & -3 \end{pmatrix}",
            r"\begin{bmatrix} -3 & 0 \\ 0 & -3 \end{bmatrix}",
        )
        self.assert_matches(r"2x^{5}", r"2x^5")
        self.assert_matches(r"-\frac{\pi}{3}", r"-\dfrac{\pi}{3}")
        self.assert_matches(r"\dfrac{\sqrt{5}}{3}", r"\frac{\sqrt{5}}{3}")
        self.assert_matches(r"\dfrac{\sqrt[3]{4}}{6}", r"\frac{\sqrt[3]{4}}{6}")
        self.assert_matches(r"32,\!000,\!000", "32,000,000")
        self.assert_matches(r"50%", r"50\%")
        self.assert_matches(r"18x^3 + 8x^2", r"(18x^3 + 8x^2)")

    def test_multiple_choice_compact_list(self) -> None:
        self.assert_matches("A,C", "AC")

    def test_prefix_and_equation_candidates(self) -> None:
        self.assert_matches(
            r"-9x^3 + 3x^2 - 2x + 2",
            r"h(x) = -9x^3 + 3x^2 - 2x + 2",
        )
        self.assert_matches(
            r"x^2 + 7x + 10",
            r"(x+2)(x+5) = x^2 + 7x + 10",
        )
        self.assert_matches(r"h^{-1}(x) = \frac{x + 5}{6}", r"y = \frac{x + 5}{6}")
        self.assert_matches(r"x = \frac{3}{2}", r"x = \dfrac{3}{2}")
        self.assert_matches("1500", r"1500 \, \text{J}")

    def test_strict_numeric_unit_percent_degree_policy(self) -> None:
        self.assert_matches(r"130725 \, \text{Pa}", "130725")
        self.assert_matches(r"3.34 \, \text{kJ}", "3.34")
        self.assert_matches(r"70\%", "70")
        self.assert_matches(r"70\%", "0.7")
        self.assert_matches("120", r"120^\circ")
        self.assert_matches(
            r"3.97 \times 10^{-19} \, \text{J}",
            r"3.97 \times 10^{-19}",
        )
        self.assert_matches(r"-11 , \text{kJ}", "-11")
        self.assert_matches(r"2.5 , \text{g/mL}", r"2.5 \, \text{g/mL}")
        self.assert_matches(r"54", r"54\%")
        self.assert_matches(r"20 , \text{cm}", "20")
        self.assert_matches(r"0.0714 , \text{J/g°C}", "0.0714")
        self.assert_matches(r"\tan 15", r"\tan{15^\circ}")

    def test_substring_safeguards(self) -> None:
        self.assert_no_match("1331_5", "1331")
        self.assert_no_match(r"\left( \dfrac{2}{3}, 1 \right) \cup (7, \infty)", r"(7, \infty)")
        self.assert_no_match(r"\dfrac{1}{512}", "512")
        self.assert_no_match(r"\begin{pmatrix} -5 \\ 9 \\ -8 \end{pmatrix}", "-5")
        self.assert_no_match(r"50\sqrt{10}", "50")
        self.assert_no_match(r"29.63", "29.6")
        self.assert_no_match(r"25 \text{ to } 30 \, \text{kJ/mol}", r"30 \text{ kJ/mol}")
        self.assert_no_match("70", "0.7")
        self.assert_no_match("10", r"10^2")
        self.assert_no_match("25", "325")
        self.assert_no_match(r"\frac{1}{3}", r"\frac{1}{3}, -\frac{5}{2}")
        self.assert_no_match("-12", "12")
        self.assert_no_match("-23", "-2")

    def test_v2_false_positive_regressions(self) -> None:
        self.assert_no_match("October 30", r"October\ 29")
        self.assert_no_match(
            "12x - 4y + 3z - 169 = 0",
            "12x - 4y + 3z - 170 = 0",
        )
        self.assert_no_match("x + y + z = 0", "2x + 3y - 4z - 15 = 0")
        self.assert_no_match(
            r"f^{-1}(x) = \frac{4 - x}{5}",
            r"f^{-1}(x) = \frac{1}{5}x + 1",
        )
        self.assert_matches(r"6.913 \times 10^{-28} \text{ m}^{-4}", "0")
        self.assert_no_match("2x - y + 3z + 8 = 0", "-2x + 4y + 2z + 10 = 0")
        self.assert_no_match("x + y - z + 1 = 0", "-10x - 10y - 10z - 100 = 0")
        self.assert_no_match(
            r"2\text{-methylbut-1-ene}",
            r"2\text{-methyl-1,2-dimethylbutene}",
        )
        self.assert_no_match(r"2\text{-methylprop-1-ene}", "2")

    def test_numeric_product_and_polynomial_factor_matches(self) -> None:
        self.assert_matches(r"\dfrac{1}{23426}", r"\dfrac{6}{53 \times 52 \times 51}")
        self.assert_matches("2xy + 10x + 20y + 100", r"(x + 10)(2y + 10)")

    @unittest.skipUnless(matcher.SYMPY_AVAILABLE, "SymPy parser is not installed")
    def test_symbolic_equivalence(self) -> None:
        self.assert_matches(r"\dfrac{-7x + 8}{6}", r"\frac{8 - 7x}{6}")
        self.assert_matches("x + y - z + 1 = 0", "-10x - 10y + 10z - 10 = 0")


    def test_additional_report_format_equivalent_cases(self) -> None:
        """Extra cases from the false-negative report that were not explicit in the old test."""
        cases = [
            (r"\text{even}", "even"),
            ("Mammalia", r"\text{Mammalia}"),
            (r"\text{orange}", "orange"),
            (r"\left[ -\dfrac{1}{3}, 1 \right]", r"\left[-\frac{1}{3}, 1\right]"),
            (r"\left(2, \frac{5\pi}{3}\right)", r"(2, \frac{5\pi}{3})"),
            (r"\left(6, \frac{3\pi}{2}, \frac{\pi}{3}\right)", r"(6, \frac{3\pi}{2}, \frac{\pi}{3})"),
            (r"\left( -\dfrac{1}{3}, \dfrac{22}{3} \right)", r"\left( -\frac{1}{3}, \frac{22}{3} \right)"),
            (r"\left( -\dfrac{5}{11}, \dfrac{12}{11} \right)", r"\left( -\frac{5}{11}, \frac{12}{11} \right)"),
            (r"(-4\sqrt{3}, -4)", r"\left( -4\sqrt{3}, -4 \right)"),
            (r"(0, 0, 3)", r"\left( 0, 0, 3 \right)"),
            (r"2 - \frac{\pi}{2}", r"2 - \dfrac{\pi}{2}"),
            (r"\dfrac{\sqrt{10}}{10}", r"\frac{\sqrt{10}}{10}"),
            (r"\frac{\sqrt{6} - \sqrt{2}}{4}", r"\dfrac{\sqrt{6} - \sqrt{2}}{4}"),
        ]
        for label, extracted in cases:
            with self.subTest(label=label, extracted=extracted):
                self.assert_matches(label, extracted)

    def test_additional_report_prefix_and_equation_candidates(self) -> None:
        cases = [
            (r"x^2 + 12x + 35", r"(x+5)(x+7) = x^2 + 12x + 35"),
            (r"x^2 + 1", r"p(x) = x^2 + 1"),
            (r"10t + 12", r"f(t) = 10t + 12"),
            (r"x^2 - 5x - 24", r"(x^2 - 5x - 24)"),
            (r"2x^2 - 2", r"q(x) = 2x^2 - 2"),
            (r"-x^5 + 4x^3 + 24x^2 + 16x + 1", r"p(x) = -x^5 + 4x^3 + 24x^2 + 16x + 1"),
            (r"2x^2 + 13x + 15", r"(2x + 3)(x + 5) = 2x^2 + 13x + 15"),
            (r"29b(5b + 1)", r"145b^2 + 29b = 29b(5b + 1)"),
        ]
        for label, extracted in cases:
            with self.subTest(label=label, extracted=extracted):
                self.assert_matches(label, extracted)

    def test_additional_report_unit_percent_degree_cases(self) -> None:
        cases = [
            (r"44\%", "44"),
            (r"4\%", "4"),
            (r"125\%", "125"),
            (r"6.4\%", "6.4"),
            (r"75\%", "75"),
            (r"10\%", "10"),
            (r"67", r"67^\circ"),
            (r"30", r"30^\circ"),
            (r"0", r"0^\circ"),
            (r"62", r"62^\circ"),
            (r"3.28 \times 10^{13} \, \text{J}", r"3.28 \times 10^{13}"),
            (r"1.875 \times 10^{-22} , \text{J}", r"1.875 \times 10^{-22}"),
            (r"7.2 \times 10^{54} \, \text{erg}", r"7.2 \times 10^{54}"),
            (r"-265.8 , \text{J/mol K}", r"-265.8 \, \text{J/mol K}"),
            (r"-56.7", r"-56.7 \, \text{kJ/mol}"),
            (r"17.55 , \text{kJ}", "17.55"),
        ]
        for label, extracted in cases:
            with self.subTest(label=label, extracted=extracted):
                self.assert_matches(label, extracted)

    def test_additional_report_benign_extra_text_matches(self) -> None:
        """Cases where the extracted answer adds only a variable/name wrapper or an equality expansion."""
        cases = [
            (r"-5 \mathbf{I}", r"M = -5I"),
            (r"-\frac{1}{2}", r"x = -\frac{1}{2}"),
            (r"\dfrac{2}{3}", r"y = \frac{2}{3}"),
            (r"\dfrac{1}{2}", r"a = \frac{1}{2}"),
            (r"(0, \infty)", r"y \in \left(0, \infty\right)"),
            (r"1 + 3i", r"z = 1 + 3i"),
            (r"-1", r"\left\{ -1 \right\}"),
        ]
        for label, extracted in cases:
            with self.subTest(label=label, extracted=extracted):
                self.assert_matches(label, extracted)

    def test_additional_report_partial_answer_no_matches(self) -> None:
        """Partial answers or longer texts that merely contain the label must not pass."""
        cases = [
            (r"4002", r"4002 + \dfrac{1}{2^{2001}}"),
            (r"44.7", r"44.72"),
            (r"10x", r"10x^2"),
            (r"\sin^2 y", r"1 - \sin^2 x - \sin^2 y"),
            (r"-1", r"-1, 1, 3"),
            (r"-1", r"-1, -5"),
            (r"x^4 - 4", r"f(x^2 - 1) = x^4 - 4x^2 + 4"),
            (r"10", r"10.75"),
            (r"10", r"10^2"),
            (r"25", r"325"),
        ]
        for label, extracted in cases:
            with self.subTest(label=label, extracted=extracted):
                self.assert_no_match(label, extracted)


if __name__ == "__main__":
    unittest.main()
