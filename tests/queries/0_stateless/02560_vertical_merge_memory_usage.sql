-- Tags: no-object-storage
drop table if exists tvm;
create table tvm (c0 UInt64, c1 UInt64, c2 UInt64, c3 UInt64, c4 UInt64, c5 UInt64, c6 UInt64, c7 UInt64, c8 UInt64, c9 UInt64, c10 UInt64, c11 UInt64, c12 UInt64, c13 UInt64, c14 UInt64, c15 UInt64, c16 UInt64, c17 UInt64, c18 UInt64, c19 UInt64, c20 UInt64, c21 UInt64, c22 UInt64, c23 UInt64, c24 UInt64, c25 UInt64, c26 UInt64, c27 UInt64, c28 UInt64, c29 UInt64, c30 UInt64, c31 UInt64, c32 UInt64, c33 UInt64, c34 UInt64, c35 UInt64, c36 UInt64, c37 UInt64, c38 UInt64, c39 UInt64, c40 UInt64, c41 UInt64, c42 UInt64, c43 UInt64, c44 UInt64, c45 UInt64, c46 UInt64, c47 UInt64, c48 UInt64, c49 UInt64, c50 UInt64, c51 UInt64, c52 UInt64, c53 UInt64, c54 UInt64, c55 UInt64, c56 UInt64, c57 UInt64, c58 UInt64, c59 UInt64, c60 UInt64, c61 UInt64, c62 UInt64, c63 UInt64, c64 UInt64, c65 UInt64, c66 UInt64, c67 UInt64, c68 UInt64, c69 UInt64, c70 UInt64, c71 UInt64, c72 UInt64, c73 UInt64, c74 UInt64, c75 UInt64, c76 UInt64, c77 UInt64, c78 UInt64, c79 UInt64, c80 UInt64, c81 UInt64, c82 UInt64, c83 UInt64, c84 UInt64, c85 UInt64, c86 UInt64, c87 UInt64, c88 UInt64, c89 UInt64, c90 UInt64, c91 UInt64, c92 UInt64, c93 UInt64, c94 UInt64, c95 UInt64, c96 UInt64, c97 UInt64, c98 UInt64, c99 UInt64, c100 UInt64, c101 UInt64, c102 UInt64, c103 UInt64, c104 UInt64, c105 UInt64, c106 UInt64, c107 UInt64, c108 UInt64, c109 UInt64, c110 UInt64, c111 UInt64, c112 UInt64, c113 UInt64, c114 UInt64, c115 UInt64, c116 UInt64, c117 UInt64, c118 UInt64, c119 UInt64, c120 UInt64, c121 UInt64, c122 UInt64, c123 UInt64, c124 UInt64, c125 UInt64, c126 UInt64, c127 UInt64, c128 UInt64, c129 UInt64, c130 UInt64, c131 UInt64, c132 UInt64, c133 UInt64, c134 UInt64, c135 UInt64, c136 UInt64, c137 UInt64, c138 UInt64, c139 UInt64, c140 UInt64, c141 UInt64, c142 UInt64, c143 UInt64, c144 UInt64, c145 UInt64, c146 UInt64, c147 UInt64, c148 UInt64, c149 UInt64, c150 UInt64, c151 UInt64, c152 UInt64, c153 UInt64, c154 UInt64, c155 UInt64, c156 UInt64, c157 UInt64, c158 UInt64, c159 UInt64, c160 UInt64, c161 UInt64, c162 UInt64, c163 UInt64, c164 UInt64, c165 UInt64, c166 UInt64, c167 UInt64, c168 UInt64, c169 UInt64, c170 UInt64, c171 UInt64, c172 UInt64, c173 UInt64, c174 UInt64, c175 UInt64, c176 UInt64, c177 UInt64, c178 UInt64, c179 UInt64, c180 UInt64, c181 UInt64, c182 UInt64, c183 UInt64, c184 UInt64, c185 UInt64, c186 UInt64, c187 UInt64, c188 UInt64, c189 UInt64, c190 UInt64, c191 UInt64, c192 UInt64, c193 UInt64, c194 UInt64, c195 UInt64, c196 UInt64, c197 UInt64, c198 UInt64, c199 UInt64, c200 UInt64, c201 UInt64, c202 UInt64, c203 UInt64, c204 UInt64, c205 UInt64, c206 UInt64, c207 UInt64, c208 UInt64, c209 UInt64, c210 UInt64, c211 UInt64, c212 UInt64, c213 UInt64, c214 UInt64, c215 UInt64, c216 UInt64, c217 UInt64, c218 UInt64, c219 UInt64, c220 UInt64, c221 UInt64, c222 UInt64, c223 UInt64, c224 UInt64, c225 UInt64, c226 UInt64, c227 UInt64, c228 UInt64, c229 UInt64, c230 UInt64, c231 UInt64, c232 UInt64, c233 UInt64, c234 UInt64, c235 UInt64, c236 UInt64, c237 UInt64, c238 UInt64, c239 UInt64, c240 UInt64, c241 UInt64, c242 UInt64, c243 UInt64, c244 UInt64, c245 UInt64, c246 UInt64, c247 UInt64, c248 UInt64, c249 UInt64, c250 UInt64, c251 UInt64, c252 UInt64, c253 UInt64, c254 UInt64, c255 UInt64, c256 UInt64, c257 UInt64, c258 UInt64, c259 UInt64, c260 UInt64, c261 UInt64, c262 UInt64, c263 UInt64, c264 UInt64, c265 UInt64, c266 UInt64, c267 UInt64, c268 UInt64, c269 UInt64, c270 UInt64, c271 UInt64, c272 UInt64, c273 UInt64, c274 UInt64, c275 UInt64, c276 UInt64, c277 UInt64, c278 UInt64, c279 UInt64, c280 UInt64, c281 UInt64, c282 UInt64, c283 UInt64, c284 UInt64, c285 UInt64, c286 UInt64, c287 UInt64, c288 UInt64, c289 UInt64, c290 UInt64, c291 UInt64, c292 UInt64, c293 UInt64, c294 UInt64, c295 UInt64, c296 UInt64, c297 UInt64, c298 UInt64, c299 UInt64) engine = MergeTree order by tuple() settings min_rows_for_wide_part = 10, min_bytes_for_wide_part=0, vertical_merge_algorithm_min_rows_to_activate=1;

insert into tvm select number + 0, number + 1, number + 2, number + 3, number + 4, number + 5, number + 6, number + 7, number + 8, number + 9, number + 10, number + 11, number + 12, number + 13, number + 14, number + 15, number + 16, number + 17, number + 18, number + 19, number + 20, number + 21, number + 22, number + 23, number + 24, number + 25, number + 26, number + 27, number + 28, number + 29, number + 30, number + 31, number + 32, number + 33, number + 34, number + 35, number + 36, number + 37, number + 38, number + 39, number + 40, number + 41, number + 42, number + 43, number + 44, number + 45, number + 46, number + 47, number + 48, number + 49, number + 50, number + 51, number + 52, number + 53, number + 54, number + 55, number + 56, number + 57, number + 58, number + 59, number + 60, number + 61, number + 62, number + 63, number + 64, number + 65, number + 66, number + 67, number + 68, number + 69, number + 70, number + 71, number + 72, number + 73, number + 74, number + 75, number + 76, number + 77, number + 78, number + 79, number + 80, number + 81, number + 82, number + 83, number + 84, number + 85, number + 86, number + 87, number + 88, number + 89, number + 90, number + 91, number + 92, number + 93, number + 94, number + 95, number + 96, number + 97, number + 98, number + 99, number + 100, number + 101, number + 102, number + 103, number + 104, number + 105, number + 106, number + 107, number + 108, number + 109, number + 110, number + 111, number + 112, number + 113, number + 114, number + 115, number + 116, number + 117, number + 118, number + 119, number + 120, number + 121, number + 122, number + 123, number + 124, number + 125, number + 126, number + 127, number + 128, number + 129, number + 130, number + 131, number + 132, number + 133, number + 134, number + 135, number + 136, number + 137, number + 138, number + 139, number + 140, number + 141, number + 142, number + 143, number + 144, number + 145, number + 146, number + 147, number + 148, number + 149, number + 150, number + 151, number + 152, number + 153, number + 154, number + 155, number + 156, number + 157, number + 158, number + 159, number + 160, number + 161, number + 162, number + 163, number + 164, number + 165, number + 166, number + 167, number + 168, number + 169, number + 170, number + 171, number + 172, number + 173, number + 174, number + 175, number + 176, number + 177, number + 178, number + 179, number + 180, number + 181, number + 182, number + 183, number + 184, number + 185, number + 186, number + 187, number + 188, number + 189, number + 190, number + 191, number + 192, number + 193, number + 194, number + 195, number + 196, number + 197, number + 198, number + 199, number + 200, number + 201, number + 202, number + 203, number + 204, number + 205, number + 206, number + 207, number + 208, number + 209, number + 210, number + 211, number + 212, number + 213, number + 214, number + 215, number + 216, number + 217, number + 218, number + 219, number + 220, number + 221, number + 222, number + 223, number + 224, number + 225, number + 226, number + 227, number + 228, number + 229, number + 230, number + 231, number + 232, number + 233, number + 234, number + 235, number + 236, number + 237, number + 238, number + 239, number + 240, number + 241, number + 242, number + 243, number + 244, number + 245, number + 246, number + 247, number + 248, number + 249, number + 250, number + 251, number + 252, number + 253, number + 254, number + 255, number + 256, number + 257, number + 258, number + 259, number + 260, number + 261, number + 262, number + 263, number + 264, number + 265, number + 266, number + 267, number + 268, number + 269, number + 270, number + 271, number + 272, number + 273, number + 274, number + 275, number + 276, number + 277, number + 278, number + 279, number + 280, number + 281, number + 282, number + 283, number + 284, number + 285, number + 286, number + 287, number + 288, number + 289, number + 290, number + 291, number + 292, number + 293, number + 294, number + 295, number + 296, number + 297, number + 298, number + 299 from numbers(20);

optimize table tvm final;

system flush logs;
-- should be about 4MB
select formatReadableSize(peak_memory_usage), * from system.part_log where table = 'tvm' and database = currentDatabase() and event_date >= today() - 1 and event_type = 'MergeParts' and peak_memory_usage > 100000000 format Vertical;

drop table tvm;
