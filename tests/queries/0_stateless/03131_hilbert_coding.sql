SELECT '----- START -----';
drop table if exists hilbert_numbers_03131;
create table hilbert_numbers_03131(
    n1 UInt32,
    n2 UInt32
)
    Engine=MergeTree()
    ORDER BY n1 SETTINGS index_granularity = 8192, index_granularity_bytes = '10Mi';

SELECT '----- CONST -----';
select hilbertEncode(133);
select hilbertEncode(3, 4);
select hilbertDecode(2, 31);

SELECT '----- 4294967296, 2 -----';
insert into hilbert_numbers_03131
select n1.number, n2.number
from numbers(pow(2, 32)-8,8) n1
    cross join numbers(pow(2, 32)-8, 8) n2
;

drop table if exists hilbert_numbers_1_03131;
create table hilbert_numbers_1_03131(
    n1 UInt64,
    n2 UInt64
)
    Engine=MergeTree()
    ORDER BY n1 SETTINGS index_granularity = 8192, index_granularity_bytes = '10Mi';

insert into hilbert_numbers_1_03131
select untuple(hilbertDecode(2, hilbertEncode(n1, n2)))
from hilbert_numbers_03131;

(
    select n1, n2 from hilbert_numbers_03131
    union distinct
    select n1, n2 from hilbert_numbers_1_03131
)
except
(
    select n1, n2 from hilbert_numbers_03131
    intersect
    select n1, n2 from hilbert_numbers_1_03131
);
drop table if exists hilbert_numbers_1_03131;

SELECT '----- END -----';
drop table if exists hilbert_numbers_03131;
