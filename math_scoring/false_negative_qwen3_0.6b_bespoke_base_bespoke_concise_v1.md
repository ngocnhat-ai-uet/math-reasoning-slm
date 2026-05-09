# Wrong-answer audit report

Total rows checked: 1545

## Summary

- likely_true_wrong: 1274
- likely_mislabeled_false_negative: 171
- probably_mislabeled_if_units_optional: 41
- partial_or_ambiguous: 29
- manual_review_maybe_extra_text: 20
- partial_answer_wrong: 5
- partial_or_overinclusive_wrong: 3
- review_long_text_contains_answer: 2

## Suggested interpretation

- `likely_mislabeled_false_negative`: nên đổi sang đúng hoặc sửa comparator/normalizer.
- `probably_mislabeled_if_units_optional`: số đúng, chỉ khác đơn vị/%/degree; quyết định theo policy chấm.
- `review_long_text_contains_answer`: output dài có chứa đáp án, gán nhãn thủ công bằng tay.
- `partial_or_ambiguous` / `manual_review_maybe_extra_text`: cần xem đề vì heuristic chỉ thấy quan hệ chứa chuỗi.


## Flagged rows


### likely_mislabeled_false_negative (171)

- `028a2b78c3` | format_equivalent | label: `\frac{10^{-2.5}}{\gamma_{\text{HCl}}}` | extracted: `\dfrac{10^{-2.5}}{\gamma_{HCl}}`
- `02cfc206d3` | format_equivalent | label: `\frac{26}{15}` | extracted: `\dfrac{26}{15}`
- `03793ea38e` | algebra_equivalent_or_prefix | label: `h^{-1}(x) = \frac{x + 5}{6}` | extracted: `y = \frac{x + 5}{6}`
- `0475436519` | algebra_equivalent_or_prefix | label: `-9x^3 + 3x^2 - 2x + 2` | extracted: `h(x) = -9x^3 + 3x^2 - 2x + 2`
- `05388eea18` | format_equivalent | label: `\text{even}` | extracted: `even`
- `061badd94b` | format_equivalent | label: `\dfrac{7}{4}` | extracted: `\frac{7}{4}`
- `109cd51c85` | format_equivalent | label: `A,C` | extracted: `AC`
- `11a6859bf1` | format_equivalent | label: `\left( \dfrac{13}{5}, \dfrac{23}{5}, 0 \right)` | extracted: `\left( \frac{13}{5}, \frac{23}{5}, 0 \right)`
- `11bf9838b1` | format_equivalent | label: `\dfrac{21}{8}` | extracted: `\frac{21}{8}`
- `11c0e4d532` | algebra_equivalent_or_prefix | label: `x^2 + 12x + 35` | extracted: `(x+5)(x+7) = x^2 + 12x + 35`
- `17e9165ab1` | algebra_equivalent_or_prefix | label: `x^2 + 1` | extracted: `p(x) = x^2 + 1`
- `18e6ef2aa9` | format_equivalent | label: `\dfrac{3}{35}` | extracted: `\frac{3}{35}`
- `19057c603a` | format_equivalent | label: `[-3, -\frac{4}{3})` | extracted: `[-3, -\dfrac{4}{3})`
- `1bf79e6469` | format_equivalent | label: `\frac{3}{2}` | extracted: `\dfrac{3}{2}`
- `1d3801b2b8` | format_equivalent | label: `-\frac{3}{2}` | extracted: `-\dfrac{3}{2}`
- `1ebb33164f` | format_equivalent | label: `\dfrac{27}{26}` | extracted: `\frac{27}{26}`
- `1f2862beee` | format_equivalent | label: `50%` | extracted: `50\%`
- `1f5e6290f5` | algebra_equivalent_or_prefix | label: `18x^3 + 8x^2` | extracted: `(18x^3 + 8x^2)`
- `2031f0b028` | format_equivalent | label: `-\frac{\pi}{3}` | extracted: `-\dfrac{\pi}{3}`
- `254b97a405` | algebra_equivalent_or_prefix | label: `10t + 12` | extracted: `f(t) = 10t + 12`
- `25c9e0b0cb` | format_equivalent | label: `\dfrac{1}{2}` | extracted: `\frac{1}{2}`
- `270d8a8b8f` | format_equivalent | label: `\left[ -\dfrac{1}{3}, 1 \right]` | extracted: `\left[-\frac{1}{3}, 1\right]`
- `27d1566c1c` | format_equivalent | label: `-\dfrac{1}{2}` | extracted: `-\frac{1}{2}`
- `29bf9408e1` | format_equivalent | label: `\frac{1}{8}` | extracted: `\dfrac{1}{8}`
- `2a5a8c861b` | format_equivalent | label: `\text{odd}` | extracted: `odd`
- `2a86ac1200` | format_equivalent | label: `\left(2, \frac{5\pi}{3}\right)` | extracted: `(2, \frac{5\pi}{3})`
- `2b7cdd1e36` | format_equivalent | label: `\dfrac{11}{12}` | extracted: `\frac{11}{12}`
- `2be6643561` | format_equivalent | label: `\frac{2}{5}` | extracted: `\dfrac{2}{5}`
- `2c9bc3e51d` | format_equivalent | label: `\dfrac{14}{15}` | extracted: `\frac{14}{15}`
- `2ca4da4a31` | format_equivalent | label: `89, 88, 85, 73, 70` | extracted: `89, 85, 88, 73, 70`
- `402308eb6d` | format_equivalent | label: `\left(6, \frac{3\pi}{2}, \frac{\pi}{3}\right)` | extracted: `(6, \frac{3\pi}{2}, \frac{\pi}{3})`
- `676bafe0c8` | algebra_equivalent_or_prefix | label: `130725 \, \text{Pa}` | extracted: `130725`
- `6891b0741f` | format_equivalent | label: `\dfrac{4}{3}` | extracted: `\frac{4}{3}`
- `68fdf07e7a` | format_equivalent | label: `-\frac{5}{4}` | extracted: `-\dfrac{5}{4}`
- `6c737cd30d` | format_equivalent | label: `Mammalia` | extracted: `\text{Mammalia}`
- `78f30ed0dd` | format_equivalent | label: `\left( -\dfrac{1}{3}, \dfrac{22}{3} \right)` | extracted: `\left( -\frac{1}{3}, \frac{22}{3} \right)`
- `7d3a100649` | algebra_equivalent_or_prefix | label: `\dfrac{-7x + 8}{6}` | extracted: `\frac{8 - 7x}{6}`
- `7d7136103e` | format_equivalent | label: `\frac{1}{3}` | extracted: `\dfrac{1}{3}`
- `7f19b54a10` | format_equivalent | label: `\frac{5\pi}{3}` | extracted: `\dfrac{5\pi}{3}`
- `812caa11f9` | format_equivalent | label: `\dfrac{2}{7}` | extracted: `\frac{2}{7}`
- `81f28d3f9a` | format_equivalent | label: `(6, -17)` | extracted: `\left( 6, -17 \right)`
- `831aeb9975` | format_equivalent | label: `\dfrac{15}{4}` | extracted: `\frac{15}{4}`
- `84724d5489` | format_equivalent | label: `\dfrac{\sqrt{5}}{3}` | extracted: `\frac{\sqrt{5}}{3}`
- `85e0d12eda` | format_equivalent | label: `-\dfrac{\sqrt{2}}{2}` | extracted: `-\frac{\sqrt{2}}{2}`
- `86d711e024` | format_equivalent | label: `\dfrac{77}{6}` | extracted: `\frac{77}{6}`
- `87be33e3b2` | format_equivalent | label: `\dfrac{7}{8}` | extracted: `\frac{7}{8}`
- `88a204d0d5` | algebra_equivalent_or_prefix | label: `1500` | extracted: `1500 \, \text{J}`
- `88a5631b37` | format_equivalent | label: `-\dfrac{\sqrt{2}}{2}` | extracted: `-\frac{\sqrt{2}}{2}`
- `8926ad62c9` | format_equivalent | label: `\dfrac{3}{8}` | extracted: `\frac{3}{8}`
- `8aa137927a` | algebra_equivalent_or_prefix | label: `x^2 - 5x - 24` | extracted: `(x^2 - 5x - 24)`
- `8b4e172546` | format_equivalent | label: `\text{orange}` | extracted: `orange`
- `8cac390cf2` | format_equivalent | label: `\dfrac{5}{7}` | extracted: `\frac{5}{7}`
- `8d1be3e0f2` | format_equivalent | label: `\dfrac{\sqrt[3]{4}}{6}` | extracted: `\frac{\sqrt[3]{4}}{6}`
- `8d86e395f5` | format_equivalent | label: `\frac{1}{169}` | extracted: `\dfrac{1}{169}`
- `8ddbeaaaab` | format_equivalent | label: `\begin{pmatrix} -3 & 0 \\ 0 & -3 \end{pmatrix}` | extracted: `\begin{bmatrix} -3 & 0 \\ 0 & -3 \end{bmatrix}`
- `9335f72d59` | format_equivalent | label: `\frac{3}{7}` | extracted: `\dfrac{3}{7}`
- `9343a84747` | format_equivalent | label: `\dfrac{9}{2}` | extracted: `\frac{9}{2}`
- `98d2144429` | algebra_equivalent_or_prefix | label: `x^2 + 7x + 10` | extracted: `(x+2)(x+5) = x^2 + 7x + 10`
- `9955aef47d` | format_equivalent | label: `32,\!000,\!000` | extracted: `32,000,000`
- `9a4ca04cde` | format_equivalent | label: `\frac{17}{35}` | extracted: `\dfrac{17}{35}`
- `9b2beb0980` | format_equivalent | label: `\left( \dfrac{8}{5}, -\dfrac{35}{2} \right)` | extracted: `\left( \frac{8}{5}, -\frac{35}{2} \right)`
- `9b68025df8` | format_equivalent | label: `\frac{\sqrt{6} - \sqrt{2}}{4}` | extracted: `\dfrac{\sqrt{6} - \sqrt{2}}{4}`
- `9c419969a8` | format_equivalent | label: `\frac{2}{7}` | extracted: `\dfrac{2}{7}`
- `9cd9d9e3cc` | format_equivalent | label: `\frac{\pi}{5}` | extracted: `\dfrac{\pi}{5}`
- `9d18ccead6` | format_equivalent | label: `\dfrac{\pi}{2}` | extracted: `\frac{\pi}{2}`
- `9d9d4698c7` | algebra_equivalent_or_prefix | label: `2x^2 - 2` | extracted: `q(x) = 2x^2 - 2`
- `9e9bd549c9` | format_equivalent | label: `x = \frac{3}{2}` | extracted: `x = \dfrac{3}{2}`
- `9fffaa40b6` | format_equivalent | label: `\frac{3}{5}` | extracted: `\dfrac{3}{5}`
- `a42ed929e6` | format_equivalent | label: `\left[ -\frac{3}{2}, \frac{2}{5} \right]` | extracted: `\left[ -\dfrac{3}{2}, \dfrac{2}{5} \right]`
- `a4a3cf2fb4` | format_equivalent | label: `\text{even}` | extracted: `even`
- `a60f9f8475` | format_equivalent | label: `-\frac{1}{7}` | extracted: `-\dfrac{1}{7}`
- `a739b231d0` | format_equivalent | label: `\dfrac{4}{3}` | extracted: `\frac{4}{3}`
- `a7f7929175` | format_equivalent | label: `\left(2, \frac{7\pi}{4}\right)` | extracted: `(2, \frac{7\pi}{4})`
- `ab3842a701` | format_equivalent | label: `-\frac{7}{2}` | extracted: `-\dfrac{7}{2}`
- `ab536d4190` | format_equivalent | label: `\dfrac{1}{57}` | extracted: `\frac{1}{57}`
- `ac7a36c735` | format_equivalent | label: `\left( -\dfrac{5}{11}, \dfrac{12}{11} \right)` | extracted: `\left( -\frac{5}{11}, \frac{12}{11} \right)`
- `ad1b61dd82` | format_equivalent | label: `\dfrac{1}{14}` | extracted: `\frac{1}{14}`
- `ad60e1a151` | format_equivalent | label: `\dfrac{16}{65}` | extracted: `\frac{16}{65}`
- `ad6d61eddd` | format_equivalent | label: `\dfrac{1}{3}` | extracted: `\frac{1}{3}`
- `ad99c8c20a` | format_equivalent | label: `2x^{5}` | extracted: `2x^5`
- `ae95fa379e` | format_equivalent | label: `-\dfrac{11}{12}` | extracted: `-\frac{11}{12}`
- `b144beb2bf` | format_equivalent | label: `-\frac{3}{4}` | extracted: `-\dfrac{3}{4}`
- `b155c01019` | format_equivalent | label: `\frac{64}{9}` | extracted: `\dfrac{64}{9}`
- `b23e348ce8` | format_equivalent | label: `\left( \dfrac{7}{9}, \dfrac{2}{9} \right)` | extracted: `\left( \frac{7}{9}, \frac{2}{9} \right)`
- `b38738d593` | format_equivalent | label: `\dfrac{17}{10}` | extracted: `\frac{17}{10}`
- `b394fdea7e` | format_equivalent | label: `\frac{63}{64}` | extracted: `\dfrac{63}{64}`
- `b481bb7672` | format_equivalent | label: `-\frac{11}{2}` | extracted: `-\dfrac{11}{2}`
- `b4852667a4` | format_equivalent | label: `\dfrac{4}{3}` | extracted: `\frac{4}{3}`
- `b7df70b1fd` | format_equivalent | label: `\dfrac{4}{13}` | extracted: `\frac{4}{13}`
- `b86726c336` | format_equivalent | label: `(-4\sqrt{3}, -4)` | extracted: `\left( -4\sqrt{3}, -4 \right)`
- `ba51c336ad` | format_equivalent | label: `\dfrac{13}{8}` | extracted: `\frac{13}{8}`
- `bbd7fc7dc9` | format_equivalent | label: `\dfrac{3}{5}` | extracted: `\frac{3}{5}`
- `c091e608fb` | format_equivalent | label: `-\dfrac{12}{5}` | extracted: `-\frac{12}{5}`
- `c286ed13fc` | format_equivalent | label: `\left( -1, \frac{35}{12} \right)` | extracted: `\left(-1, \dfrac{35}{12}\right)`
- `c32592e8e8` | format_equivalent | label: `\dfrac{11}{21}` | extracted: `\frac{11}{21}`
- `c6e1defa21` | format_equivalent | label: `\frac{13}{4}` | extracted: `\dfrac{13}{4}`
- `c8ae5f2316` | format_equivalent | label: `\dfrac{56}{25}` | extracted: `\frac{56}{25}`
- `c957fba06c` | format_equivalent | label: `(8, \frac{\pi}{3}, \frac{2\pi}{3})` | extracted: `(8, \dfrac{\pi}{3}, \dfrac{2\pi}{3})`
- `ca5a7ff20c` | format_equivalent | label: `(-3, \frac{1}{14})` | extracted: `\left( -3, \frac{1}{14} \right)`
- `cb22be3c40` | algebra_equivalent_or_prefix | label: `2\text{-methylbut-1-ene}` | extracted: `2\text{-methyl-1,2-dimethylbutene}`
- `d0d717e417` | format_equivalent | label: `\dfrac{16}{3}` | extracted: `\frac{16}{3}`
- `d161e99235` | format_equivalent | label: `\left(2, \frac{11\pi}{8}\right)` | extracted: `\left(2, \dfrac{11\pi}{8}\right)`
- `d2c9ecdac2` | format_equivalent | label: `\dfrac{1}{20}` | extracted: `\frac{1}{20}`
- `d51f81e083` | format_equivalent | label: `2 - \frac{\pi}{2}` | extracted: `2 - \dfrac{\pi}{2}`
- `d531919c94` | format_equivalent | label: `-\dfrac{\sqrt{3}}{2}` | extracted: `-\frac{\sqrt{3}}{2}`
- `d53e5a33b4` | format_equivalent | label: `-\frac{1}{2}` | extracted: `-\dfrac{1}{2}`
- `da16e9f7bf` | format_equivalent | label: `(-\infty, -\dfrac{1}{5})` | extracted: `(-\infty, -\frac{1}{5})`
- `da7c50f285` | format_equivalent | label: `-\dfrac{4}{5}` | extracted: `-\frac{4}{5}`
- `da94cefa89` | format_equivalent | label: `\frac{61}{243}` | extracted: `\dfrac{61}{243}`
- `e11b9e4422` | format_equivalent | label: `\text{odd}` | extracted: `odd`
- `e408efe91a` | format_equivalent | label: `\dfrac{6}{11}` | extracted: `\frac{6}{11}`
- `e4a1212e8b` | format_equivalent | label: `(0, \frac{1}{4})` | extracted: `\left(0, \dfrac{1}{4}\right)`
- `e5fd6d2e82` | algebra_equivalent_or_prefix | label: `2\text{-methylprop-1-ene}` | extracted: `2`
- `e7c3b7eeab` | algebra_equivalent_or_prefix | label: `-x^5 + 4x^3 + 24x^2 + 16x + 1` | extracted: `p(x) = -x^5 + 4x^3 + 24x^2 + 16x + 1`
- `e83afffd72` | format_equivalent | label: `\dfrac{207}{110}` | extracted: `\frac{207}{110}`
- `eba705905e` | format_equivalent | label: `-\dfrac{\sqrt{7}}{4}` | extracted: `-\frac{\sqrt{7}}{4}`
- `ec8c5a327c` | format_equivalent | label: `\dfrac{63}{64}` | extracted: `\frac{63}{64}`
- `edeaafe82b` | format_equivalent | label: `\text{odd}` | extracted: `odd`
- `ef689c4025` | format_equivalent | label: `(0, 0, 3)` | extracted: `\left( 0, 0, 3 \right)`
- `efb984f7eb` | format_equivalent | label: `\frac{9}{10}` | extracted: `\dfrac{9}{10}`
- `f92f158f2f` | format_equivalent | label: `-\frac{1}{2}` | extracted: `-\dfrac{1}{2}`
- `fabdcccb08` | format_equivalent | label: `\frac{21}{8}` | extracted: `\dfrac{21}{8}`
- `fb0bc9dc2d` | format_equivalent | label: `\frac{1}{4}` | extracted: `\dfrac{1}{4}`
- `fc0cd1138b` | format_equivalent | label: `\dfrac{\sqrt{10}}{10}` | extracted: `\frac{\sqrt{10}}{10}`
- `fcd9b97b7b` | format_equivalent | label: `\dfrac{1}{2187}` | extracted: `\frac{1}{2187}`
- `fea9b7dff6` | format_equivalent | label: `\frac{1}{4}` | extracted: `\dfrac{1}{4}`
- `fef790817e` | format_equivalent | label: `\dfrac{5}{16}` | extracted: `\frac{5}{16}`
- `ff3af1754a` | format_equivalent | label: `\frac{3}{2}` | extracted: `\dfrac{3}{2}`
- `ff5453262c` | format_equivalent | label: `\dfrac{3}{5}` | extracted: `\frac{3}{5}`

### probably_mislabeled_if_units_optional (41)

- `0bbc9765de` | unit_percent_degree_only | label: `3.34 \, \text{kJ}` | extracted: `3.34`
- `18d5d8b799` | unit_percent_degree_only | label: `-11 , \text{kJ}` | extracted: `-11`
- `1d7b0f2fbf` | unit_percent_degree_only | label: `67` | extracted: `67^\circ`
- `231d137675` | unit_percent_degree_only | label: `70\%` | extracted: `70`
- `262d788d6d` | unit_percent_degree_only | label: `44\%` | extracted: `44`
- `2db30ef553` | unit_percent_degree_only | label: `4\%` | extracted: `4`
- `37616a32a3` | unit_percent_degree_only | label: `125\%` | extracted: `125`
- `44c4ffb151` | unit_percent_degree_only | label: `-21.7 , \text{kJ/mol}` | extracted: `-21.7`
- `46344b94e8` | unit_percent_degree_only | label: `3.97 \times 10^{-19} \, \text{J}` | extracted: `3.97 \times 10^{-19}`
- `479963f6cd` | unit_percent_degree_only | label: `120` | extracted: `120^\circ`
- `479b25b70d` | unit_percent_degree_only | label: `20\%` | extracted: `20`
- `4e841b89ea` | unit_percent_degree_only | label: `20\%` | extracted: `20`
- `576b74029f` | unit_percent_degree_only | label: `6.4\%` | extracted: `6.4`
- `5dbc4df93f` | unit_percent_degree_only | label: `3.28 \times 10^{13} \, \text{J}` | extracted: `3.28 \times 10^{13}`
- `79ad80b399` | unit_percent_degree_only | label: `75\%` | extracted: `75`
- `7b42da9ecb` | unit_percent_degree_only | label: `30` | extracted: `30^\circ`
- `866f8b2aeb` | unit_percent_degree_only | label: `73\%` | extracted: `73`
- `8d9c8640b2` | unit_percent_degree_only | label: `35` | extracted: `35^\circ`
- `955f5bad27` | unit_percent_degree_only | label: `10\%` | extracted: `10`
- `9d368facf5` | unit_percent_degree_only | label: `2.5 , \text{g/mL}` | extracted: `2.5 \, \text{g/mL}`
- `a800c43817` | unit_percent_degree_only | label: `\tan 15` | extracted: `\tan{15^\circ}`
- `af6921d982` | unit_percent_degree_only | label: `18\%` | extracted: `18`
- `b218c51b14` | unit_percent_degree_only | label: `56\%` | extracted: `56`
- `b2ca918a61` | unit_percent_degree_only | label: `1.875 \times 10^{-22} , \text{J}` | extracted: `1.875 \times 10^{-22}`
- `b3806c5f25` | unit_percent_degree_only | label: `32\%` | extracted: `32`
- `b757e4cdf6` | unit_percent_degree_only | label: `0.0714 , \text{J/g°C}` | extracted: `0.0714`
- `bfa0116716` | unit_percent_degree_only | label: `0` | extracted: `0^\circ`
- `c02293be06` | unit_percent_degree_only | label: `-265.8 , \text{J/mol K}` | extracted: `-265.8 \, \text{J/mol K}`
- `c2f31d1081` | unit_percent_degree_only | label: `40\%` | extracted: `40`
- `c533ab5b5b` | unit_percent_degree_only | label: `62` | extracted: `62^\circ`
- `c72fa96878` | unit_percent_degree_only | label: `60` | extracted: `60^\circ`
- `cf5406fdf7` | unit_percent_degree_only | label: `10\%` | extracted: `10`
- `d416ead5da` | unit_percent_degree_only | label: `27\%` | extracted: `27`
- `d42132b83c` | unit_percent_degree_only | label: `17.55 , \text{kJ}` | extracted: `17.55`
- `d561250ad2` | unit_percent_degree_only | label: `50\%` | extracted: `50`
- `d72156a27f` | unit_percent_degree_only | label: `7.2 \times 10^{54} \, \text{erg}` | extracted: `7.2 \times 10^{54}`
- `d958711e06` | unit_percent_degree_only | label: `8\%` | extracted: `8`
- `dcb00f803d` | unit_percent_degree_only | label: `54` | extracted: `54\%`
- `df4c81c282` | unit_percent_degree_only | label: `78\%` | extracted: `78`
- `e2c9309669` | unit_percent_degree_only | label: `-56.7` | extracted: `-56.7 \, \text{kJ/mol}`
- `ff637105a1` | unit_percent_degree_only | label: `20 , \text{cm}` | extracted: `20`

### partial_or_ambiguous (29)

- `005a6383d8` | label_contains_extracted | label: `1331_5` | extracted: `1331`
- `01931e79f7` | label_contains_extracted | label: `\left( \dfrac{2}{3}, 1 \right) \cup (7, \infty)` | extracted: `(7, \infty)`
- `033d89c238` | label_contains_extracted | label: `\dfrac{1}{512}` | extracted: `512`
- `1622212a1d` | label_contains_extracted | label: `-253 \, \text{J}` | extracted: `253`
- `1d0c84267a` | label_contains_extracted | label: `20412_5` | extracted: `20412`
- `22f650e97f` | label_contains_extracted | label: `1242_6` | extracted: `1242`
- `3f16987f28` | label_contains_extracted | label: `-23` | extracted: `-2`
- `470d8f66d7` | label_contains_extracted | label: `1323_4` | extracted: `1323`
- `489e7956db` | label_contains_extracted | label: `\dfrac{243}{1024}` | extracted: `243`
- `50a6e3b8cb` | label_contains_extracted | label: `\left( -\dfrac{3}{2}, \dfrac{9}{4} \right)` | extracted: `-\frac{3}{2}`
- `562c53f62f` | label_contains_extracted | label: `-34` | extracted: `34`
- `5fec58cffc` | label_contains_extracted | label: `50000` | extracted: `5000`
- `682e6dad39` | label_contains_extracted | label: `-10` | extracted: `-1`
- `7113e1f5b0` | label_contains_extracted | label: `\begin{pmatrix} -5 \\ 9 \\ -8 \end{pmatrix}` | extracted: `-5`
- `76d08b1ed7` | label_contains_extracted | label: `Anolis\ carolinensis` | extracted: `\text{Anolis}`
- `7a6ab1c850` | label_contains_extracted | label: `50\sqrt{10}` | extracted: `50`
- `7d276fc872` | label_contains_extracted | label: `-\dfrac{\pi}{6}` | extracted: `\frac{\pi}{6}`
- `8f5efd0084` | label_contains_extracted | label: `29.63` | extracted: `29.6`
- `9320006f2f` | label_contains_extracted | label: `\dfrac{10}{7}` | extracted: `10`
- `9c719a020e` | label_contains_extracted | label: `43_5` | extracted: `43`
- `a0c1a3ee57` | label_contains_extracted | label: `-10` | extracted: `-1`
- `a136acea32` | label_contains_extracted | label: `y = -\frac{10}{3}` | extracted: `-\dfrac{10}{3}`
- `a195f2a76a` | label_contains_extracted | label: `10\sqrt{3}` | extracted: `10`
- `a888fb7ffd` | label_contains_extracted | label: `776_8` | extracted: `776`
- `ab850ff986` | label_contains_extracted | label: `25 \text{ to } 30 \, \text{kJ/mol}` | extracted: `30 \text{ kJ/mol}`
- `b366e5014f` | label_contains_extracted | label: `\left[ \dfrac{\pi}{4}, \dfrac{3\pi}{4} \right]` | extracted: `\frac{3\pi}{4}`
- `b665ead18c` | label_contains_extracted | label: `-12` | extracted: `12`
- `e515a948b7` | label_contains_extracted | label: `\begin{pmatrix} 1 & 2500 \\ 0 & 1 \end{pmatrix}` | extracted: `2500`
- `f0faa3e5c7` | label_contains_extracted | label: `\dfrac{9}{1100}` | extracted: `\dfrac{9}{110}`

### manual_review_maybe_extra_text (20)

- `238e0f6b19` | extracted_contains_label | label: `-5 \mathbf{I}` | extracted: `M = -5I`
- `3533b1f337` | extracted_contains_label | label: `-\frac{1}{2}` | extracted: `x = -\frac{1}{2}`
- `50285a3206` | extracted_contains_label | label: `\dfrac{2}{3}` | extracted: `y = \frac{2}{3}`
- `5af2c24fdf` | extracted_contains_label | label: `\frac{1}{3}` | extracted: `\frac{1}{3}, -\frac{5}{2}`
- `66ec88af4f` | extracted_contains_label | label: `4002` | extracted: `4002 + \dfrac{1}{2^{2001}}`
- `6a1c3b90e1` | extracted_contains_label | label: `\dfrac{1}{2}` | extracted: `a = \frac{1}{2}`
- `70b96436bb` | extracted_contains_label | label: `44.7` | extracted: `44.72`
- `7a4846563e` | extracted_contains_label | label: `(0, \infty)` | extracted: `y \in \left(0, \infty\right)`
- `9ceeb141e2` | extracted_contains_label | label: `10x` | extracted: `10x^2`
- `9f96dbb25f` | extracted_contains_label | label: `\sin^2 y` | extracted: `1 - \sin^2 x - \sin^2 y`
- `ad78a08a1f` | extracted_contains_label | label: `-1` | extracted: `-1, 1, 3`
- `bc2ade9c5c` | extracted_contains_label | label: `x^4 - 4` | extracted: `f(x^2 - 1) = x^4 - 4x^2 + 4`
- `bced319d23` | extracted_contains_label | label: `10` | extracted: `10.75`
- `c1f18a80af` | extracted_contains_label | label: `1 + 3i` | extracted: `z = 1 + 3i`
- `c57116b51d` | extracted_contains_label | label: `-1` | extracted: `-1, -5`
- `e1cc63f6e3` | extracted_contains_label | label: `29b(5b + 1)` | extracted: `145b^2 + 29b = 29b(5b + 1)`
- `e6b9722baf` | extracted_contains_label | label: `-1` | extracted: `\left\{ -1 \right\}`
- `ebc26be2da` | extracted_contains_label | label: `10` | extracted: `10^2`
- `f1b6b060a4` | extracted_contains_label | label: `25` | extracted: `325`
- `fcbe8e6ff0` | extracted_contains_label | label: `2x^2 + 13x + 15` | extracted: `(2x + 3)(x + 5) = 2x^2 + 13x + 15`

### review_long_text_contains_answer (2)

- `219b0e93c1` | long_text_contains_label | label: `\Delta G = \Delta H - T\Delta S` | extracted: `\text{The Gibbs free energy change for denaturation is calculated using } \Delta G = \Delta H - T\Delta S. \text{ A negative } \Delta G \text{ indicates the native state is more stable, and the protein is less likely to denature.}`
- `51278acb68` | long_text_contains_label | label: `2.87` | extracted: `4.74 \text{ (pKa of acetic acid)}, \quad \text{pH of 0.1 M acetic acid is approximately 2.87}`