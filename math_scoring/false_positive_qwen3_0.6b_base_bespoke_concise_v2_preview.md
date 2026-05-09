# v2 false positive review

## Certain false positives

| question_id | label | extracted_answer | reason |
|---|---|---|---|
| `003ee3537d` | `October 30` | `October\ 29` | Ngày khác nhau; không nên normalize 29 thành 30. |
| `6041ac8979` | `12x - 4y + 3z - 169 = 0` | `12x - 4y + 3z - 170 = 0` | Hai mặt phẳng khác nhau, hằng số lệch 1. |
| `624c07f481` | `x + y + z = 0` | `2x + 3y - 4z - 15 = 0` | Hai phương trình mặt phẳng không tỉ lệ hệ số. |
| `8de1867de6` | `f^{-1}(x) = (4 - x)/5` | `f^{-1}(x) = x/5 + 1` | Hai hàm khác nhau: (4-x)/5 != x/5+1. |
| `986a365b2e` | `2x - y + 3z + 8 = 0` | `-2x + 4y + 2z + 10 = 0` | Hai mặt phẳng không tỉ lệ hệ số. |
| `bf96097836` | `x + y - z + 1 = 0` | `-10x - 10y - 10z - 100 = 0` | Nếu nhân label với -10 sẽ là -10x-10y+10z-10=0, không khớp extracted. |
| `cb22be3c40` | `2-methylbut-1-ene` | `2-methyl-1,2-dimethylbutene` | Tên hợp chất khác; có lẽ bị match substring/token quá lỏng. |
| `e5fd6d2e82` | `2-methylprop-1-ene` | `2` | Extracted chỉ là substring "2", không phải tên hợp chất. |
