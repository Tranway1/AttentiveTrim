import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# local_model_path = '/home/gridsan/cliu/hf/Mistral-7B-Instruct-v0.2'
local_model_path = '/home/gridsan/cliu/hf/dbrx-instruct/'
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(local_model_path, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

input_text = '''
    Here is a paper snippet:
A Deep Dive into Common Open Formats for Analytical DBMSs
Chunwei Liu MIT CSAIL chunwei@csail.mit.edu
ABSTRACT
Anna Pavlenko Microsoft annapa@microsoft.com
Matteo Interlandi Microsoft mainterl@microsoft.com
Brandon Haynes Microsoft brhaynes@microsoft.com
This paper evaluates the suitability of Apache Arrow, Parquet, and ORC as formats for subsumption in an analytical DBMS. We sys- tematically identify and explore the high-level features that are important to support efficient querying in modern OLAP DBMSs and evaluate the ability of each format to support these features. We find that each format has trade-offs that make it more or less suitable for use as a format in a DBMS and identify opportunities to more holistically co-design a unified in-memory and on-disk data representation. Our hope is that this study can be used as a guide for system developers designing and using these formats, as well as provide the community with directions to pursue for improving these common open formats.
PVLDB Reference Format:
Chunwei Liu, Anna Pavlenko, Matteo Interlandi, and Brandon Haynes. A Deep Dive into Common Open Formats for Analytical DBMSs . PVLDB, 16(11): 3044 - 3056, 2023.
doi:10.14778/3611479.3611507
PVLDB Artifact Availability:
The source code, data, and/or other artifacts have been made available at https://github.com/Tranway1/ColumnarFormatsEval.
1 INTRODUCTION
Over the last decade, a number of new common and open formats have been proposed that purport to improve performance and ease interoperation across OLAP DBMSs and storage layers such as data lakes [49]. Today, storage formats such as Parquet [12] and ORC [11] are the cornerstone reference architectures for cloud- scale data warehousing systems [14]. At the same time, the in- memory format Apache Arrow [8] is widely considered to be the default means of interoperation across different data systems [50], and several systems are even exploring how to leverage it end-to- end [18]. Each of these in-memory and storage formats attempt to minimize disk, memory, and IO costs, and each applies a wide variety of optimizations to maximize analytic read performance.
Though the benefits for common open formats are now well established [29], there has been less exploration of the relative benefits of these formats for direct subsumption in an analytical DBMS as a native format. This is despite the robust discussion in database community about the relative merits of these formats for this purpose (e.g., [3, 43, 58]). One reason for this is that each for- mat makes design choices that optimize for its use as a common
This work is licensed under the Creative Commons BY-NC-ND 4.0 International License. Visit https://creativecommons.org/licenses/by-nc-nd/4.0/ to view a copy of this license. For any use beyond those covered by this license, obtain permission by emailing info@vldb.org. Copyright is held by the owner/author(s). Publication rights licensed to the VLDB Endowment.
Proceedings of the VLDB Endowment, Vol. 16, No. 11 ISSN 2150-8097. doi:10.14778/3611479.3611507
and open format, and these choices often conflict with longstand- ing analytical DBMS techniques. For example, we increasingly see a push to directly leverage Arrow as an end-to-end, in-memory format within a DBMS [37]. At the same time, DBMS in-memory columnar formats typically encode data [4, 20] to minimize space and reduce memory requirements. However, Apache Arrow by de- fault provides no encoding support, leaving it at odds with typical DBMS design. On the other hand, on-disk formats such as Parquet arrange data in a form that is much closer to that found in modern columnar DBMSs (e.g., by employing run-length encoding mixed with dictionary encoding and bit packing). However, Parquet is widely leveraged only as storage format and exposes no dedicated in-memory representation. Instead, developers bring Parquet data into memory and convert it to the Arrow format, which, as stated above, is suboptimal for a columnar DBMS. Finally, a format such as ORC at first glance appears to offer the best of both worlds, since it provides both an efficiently encoded data format and a related in-memory representation. Nevertheless, Arrow and Parquet are considered the standard nowadays because of their popularity in terms of activity in open-source projects and support from big data frameworks and large-scale query providers [30].
Given this environment, the goal of this paper is to evaluate these three formats, explore their trade-offs, and evaluate their perfor- mance as candidates for direct subsumption in an analytical DBMS. Three main challenges exist in subsuming an in-memory format such as Arrow or traditionally on-disk formats such as Parquet or ORC. First, a DBMS needs to be able to efficiently (de)serialize and (de)compress on-disk data to and from an in-memory representa- tion. For this, efficiency directly depends on format’s compression ratio, decompression speed, and transcoding performance. These trade-offs can be subtle and the line between an “in-memory” or
“on-disk” format is often blurry. For example, in some cases a DBMS could improve performance by writing Arrow to disk or directly operating on Parquet in memory, avoiding transcoding costs and taking advantage of the features offered by each format.
Second, prior work has established that it is highly advantageous for a DBMS to “push down” computation as far as possible (e.g., to disk coprocessors or into the compressed domain) and to do so over as many data types as possible [24, 56]. Computation pushdown subsumes a number of related techniques. Column pruning and data skipping respectively enable a DBMS to avoid decompressing columns or rows that do not contain data relevant to a query answer (e.g., when executing a range query by skipping data regions that do not contain data within the range). Techniques such as direct querying enable a DBMS to retrieve query answers without an expensive decode or decompression step [5, 6, 59]. As we show in Table 1, the ability for common in-memory and on-disk formats to support these techniques is uneven; to maximize performance a DBMS should optimize for the resulting trade-offs.
 3044
  Finally, to maximize performance, modern DBMSs leverage mod- ern techniques such as vectorized execution (e.g., SIMD) [27, 31, 33, 38, 40, 55] and query compilation [19]. Here again the ability for a format to support these techniques is uneven. For example, Parquet can vectorize operations over some query types whereas Arrow is inherently unencoded and more amenable to vectorized execution.
In summary, in this paper we present the first detailed, empirical evaluation of three popular and increasingly-adopted formats and evaluate their suitability to be used as a native format in a DBMS. Our hope is that this study can be used as a guide for system devel- opers using these formats, as well as provide the community with directions to pursue for improving these common open formats.
Columnar Table
Metadata
2
Our contributions include:
• Wesuccinctlysummarizethedesignnuancesanddistinc- tions of three widely-adopted open columnar formats: Apache Arrow, Parquet and ORC (Section 3).
• Wesystematicallyidentifyandexplorethehigh-levelfea- tures that are important to support efficient querying in modern OLAP DBMSs (Section 4).
• Foreachformat,weevaluateitsabilitytosupportefficient encoding, compression, and transcoding (Section 5) for both real-world, synthetic datasets, and various data types.
• Webenchmarktheabilityofeachformattosupportselect- project (SP) operations found near the leaves of query plans. We evaluate these in isolation (Section 6) and in combina- tion (Section 7) using TPC-DS query plan fragments and over various data types.
• Weevaluatetheabilityforeachformattotakeadvantage of recent trends such as vectorization, query compilation, and direct querying (Section 8).
• We identify key opportunities to holistically co-design a unified in-memory and on-disk data representation.
BACKGROUND: COMPRESSION AND DATA
ENCODING
Figure 1: Columnar format layout.
became common, columnar storage formats began to predominate [6]. Columnar databases store data of the same type together [52], allowing systems to leverage lightweight compression, also referred to as encoding [5]. Encoding methods such as dictionary encoding, run-length encoding, and bit-packed encoding are typically designed to compress a specific type of data, enabling efficient compres- sion and better record-level access relative to the general-purpose compression approaches described in the previous section.
Some encoding methods also support direct querying and data skipping to improve query performance [5, 33, 39]. Systems such as Redshift [26] and SQL Server [35] support many lightweight compression approaches that reduce the storage cost of data; at the same time, they apply compression-specific optimizations to improve query execution performance. Previous research [16, 32, 40, 41] has also demonstrated that, for specific datasets, good encoding achieves a comparable compression ratio with far fewer CPU cycles than does byte-oriented compression algorithms.
We next give an overview of several popular encoding algorithms referred to in this paper and highlight their applicable scenarios. Bit-Packed Encoding (BP) works on numerical data. It finds the minimal number of bits needed to represent values and removes superfluous leading zeros. It works best when the target numbers have similar bit-width.
Dictionary Encoding (DICT) works on all data types. It encodes each distinct entry with an integer key and bit-packs the integers. Dictionary encoding works best when the dataset has small cardi- nality and many repetitions. Queries on dictionary encoded data can be applied either on the fully decoded data or directly in the encoded domain after query rewriting using dictionary translation. Run-Length Encoding (RLE) works on data with many consec- utive repetitions. It replaces a run of the same value with a pair consisting of the value and how many times it is repeated. Hybrid Encodings are derived from the above encoding tech- niques. Dictionary run-length encoding (DICT-RLE) applies RLE on the dictionary encoded keys to further compress data. Bit-packed and run-length hybrid encoding is used as a default implementa- tion for the Parquet RLE encoder. Hybrid encoding usually achieves better compression performance at the cost of performance.
3 COLUMNAR OPEN FORMATS
In big data environments today, there are many optimized data formats for columnar data storage and computation, as we show in Table 1. Interestingly, these data formats share the same basic
Data systems employ compression algorithms to reduce on-disk or in-memory data sizes and improve bandwidth utilization [25]. Conventional compression has traditionally focused on minimizing file size. This focus on size alone, while appropriate for storage, overlooks DBMS query execution performance [33, 46, 51]. Con- versely, in an analytical columnar DBMS, compressed size is usually balanced with the ability to query directly on the compressed data.
2.1 Compression
Because of their generality, byte-oriented compression techniques (e.g., Gzip [17], Snappy [23], and Zlib [22]) are widely used to re- duce data size [5, 46]. They treat the input values as a byte stream and compress them sequentially. Byte-oriented compression is ap- plicable to all data types and, in general, exhibits good compression ratios [33]. However, these methods are computationally inten- sive [5]. A data block needs to be fully decompressed before indi- vidual values can be accessed. This often introduces unnecessary overhead for query execution.
2.2 Encoding
For decades, many data engines used row-oriented storage formats for OLTP query workloads [3]. As more complex OLAP workloads
3045
     Row Batch 1
Chunked Chunked ... Chunked Column 1 Column 2 Column m
 Row Batch 2
  ...
   Row Batch n

Table 1: A comparison of the features found in common open columnar data formats.
  Arrow Feather Parquet ORC
Encoding Methods
DICT
DICT
DICT(-RLE), RLE, BP, Delta, etc. DICT, RLE, BP, Delta
Compression Codecs
None
Zstd, LZ4
Gzip, Snappy, Zstd, LZ4, (LZO) Snappy, Zlib, LZ4
Skipping
Chunk-level None Record-level Chunk-level
Direct Query
None None None None
Primary Purpose
In-Memory Compute On-Disk Storage On-Disk Storage On-Disk Storage
Representative Systems
Dremio, Spark, Pandas, etc. Pandas
Spark, Hive, Presto, etc. Hive, Presto, etc.
       Dictionary Page
    Index data column n:
Min/Max values. Row position. Block offsets. Bloom filter.
Row data column n with Integer:
Present (non-null) bit stream. Integers encoded as RLE/Bitpacking. /////////////////////
Row data column n with String: Present (non-null) bit stream. Dictionary data (bytes).
Dictionary length (RLE).
Encoded row data (RLE).
   Row Batch x
Zone Map
 Data Page 1
    Chunked Column 1
Chunked Column 2
...
Chunked Column m
     ...
      Data Page i
  Figure 2: A Parquet row batch.
Table 2: Column format name convention mapping.
Figure 3: An ORC row batch.
  Arrow Parquet ORC
Row Batch
Record Batch Row Group Stripe
Chunked Column
Chunked Array Column Chunk Row Column
to iterate over. Arrow also defines a binary serialization protocol for converting a collection of row batches that can be used for messaging, interprocess communication (IPC), and writing blobs into storage. Deserializing an Arrow blob has effectively zero cost.
Closely related to the Arrow format, Arrow Feather [10] is a column-oriented binary disk-based format, leveraging the same IPC as the in-memory Arrow format. Additionally, Feather adds dic- tionary encoding (for strings) and compression (Zstd, LZ4). Datasets stored in Arrow Feather are loaded in-memory as Arrow Tables.
3.3 Parquet
Parquet [12] is a columnar-oriented storage format inspired by the nested data storage format outlined in Google Dremel [44]. Parquet integrates many efficient compression and encoding approaches to achieve space-efficiency. A Parquet file is structured almost exactly as described in Section 3.1; however, as illustrated in Figure 2, each column chunk is partitioned into a dictionary page and series of data pages. Its file footer additionally contains zone maps (e.g., min, max, and number of NULLs) at the row batch, chunked column, and data page level. This enables efficient data skipping. Row batches have a recommended size of 512-1024 MB. Parquet applies dictionary encoding per data page and falls back to plain encoding when a dictionary grows larger than a predefined threshold. Parquet is designed to be space and IO-efficient at the expense of CPU utilization for decoding. It does not provide any data structures for in-memory computing.
3.4 ORC
Optimized Row Columnar (ORC) [11] is a storage format designed for read-heavy analytical workloads. ORC files are organized as in Figure 1 where the default row batch size is 250 MB. Differently than Parquet, as is shown in Figure 3, ORC organizes columns into an index that contains min/max values, bloom filters, etc., and row data with a present bit-vector indicating NULL entries. The chunked columns in the row data are formatted based on the encoding type.
ORC exposes a corresponding in-memory format, which contains a row-level index and NULL bit-vector data structures for fast querying and NULL checks. ORC supports dictionary encoding (at the row batch level) for string data. Similar to Parquet, ORC falls back to plain encoding when the number of distinct values is greater than a threshold (e.g., for Hive, 80% of the records).
   underlying design. Therefore, we begin by providing a “generic” architecture that summarizes the substantial commonality in mod- ern columnar data format design (Section 3.1). Then, we drill into the idiosyncrasies found in Arrow, Parquet and ORC (Sections 3.2, 3.3, and 3.4, respectively). To ease the parsing for non-familiar read- ers, we take the liberty of providing a unified naming convention. Table 2 has the mapping between our naming and each format.
3.1 Open Columnar Formats 101
Figure 1 summarizes a generic columnar format design. Columnar storage formats physically arrange data such that all the records belonging to the same column are stored sequentially. To achieve better data access at scale, columnar formats partition columns into chunks. Chunked columns are not created arbitrarily; instead, row- level alignment is attained by first splitting a table horizontally into row batches where, within each batch, rows are then partitioned into column chunks. Metadata about the row batches (e.g., their location, number, length, compression algorithm, etc.) are stored either into the footer or in the preamble of the file.
3.2 Apache Arrow (Feather)
Apache Arrow [8] is a columnar data structure supporting effi- cient in-memory computing. Arrow can represent both flat and hierarchical data. Arrow is designed to be complementary to on- disk columnar data formats such as Parquet and ORC, and in fact it shares with them the same design depicted in Figure 1. On-disk data files are decompressed, decoded, and loaded into Arrow in-memory columnar arrays. Each row batch has a default size of 64K rows. Arrow column chunks have a present bit-vector signaling whether a value is null (or not), and, for strings, optionally a dictionary.
The Arrow columnar format has some compelling properties: random access is O(1) for entries in the same chunked column, and each value cells are sequential in memory, so it’s efficient
Row Batch x
Index data Index Col 1 Index Col 2 ... Index Col n
Row data Chunked Col 1 Chunked Col 2 ... Chunked Col n Row Batch Footer
  3046

Table 3: Default encoding by format and data type. Parquet uses DICT as default in the latest C++ API, while DICT-RLE was used in its legacy Java API. Int-RLE refers to the encoding where the decimal value is scaled to an integer and then encoded with RLE encoding.
using a predicate. We evaluate the costs of each format when
applying simple data access operations (Section 6).
4. End-to-end evaluation over subexpressions. Since we care about the performance of each format when evaluating analytical queries, we explore the performance of each format over a set of
query subexpressions drawn from TPC-DS (Section 7).
5. Advanced features. Given the many trade-offs baked into each format, we explore the extent to which we can extend them to support novel features such as computation pushdown into
the encoded domain and hardware acceleration (Section 8).
To summarize our findings, in Table 4 we show the structure of our experiments and the overall best format for each dimension.
Setup. All experiments are performed on an Azure Standard D8s v3 (8 vCPUs, 32 GiB memory), premium SSD LRS, and Ubuntu 18.04. We test Apache Arrow 5.0.0, ORC 1.7.2, and the Apache Parquet Java API version 1.9.0. Where needed, we use the Apache Arrow C++ library to write in-memory Arrow tables to disk. We perform exper- iments using (i) the TPC-DS dataset at scale 10, (ii) the Join Order Benchmark (JOB) [1], (iii) the Public BI Benchmark (BI) [2], and (iv) real-world datasets drawn from public data sources including GIS, machine learning, and financial datasets (CodecDB) [33]. For all the experiments, we report numbers when the system caches are cold by default. For selected experiments we also report numbers when caches have been warmed up, i.e., to simulate frequently accessed datasets. Unless stated otherwise, we use each format’s default settings. Different results could certainly be obtained if dataset- specific parameter tuning were applied to each format. However, such fine-grained configuration tuning is beyond the scope of the paper and left as future work.
5 COMPRESSION AND TRANSCODING
In this section, we evaluate the compression performance of Arrow, Parquet, and ORC (Section 5.1) and the related costs for transcoding data from compressed to in-memory formats (Section 5.2).
5.1 Encoding & Compression Performance
We first explore the compression performance of each format through three sets of experiments. In the first experiment (Section 5.1.1) we evaluate how each format’s supported encodings perform over a set of real-world datasets. The last two experiments leverage TPC-DS [45] to illustrate the performance of each format when compres- sion is applied on top of encodings. For the synthetic experiments on TPC-DS, we begin by evaluating compression algorithms over the full dataset (Section 5.1.2), and then explore how compression performance varies by data type (Section 5.1.3).
5.1.1 Encoding Performance over Real-World Datasets. In this ex- periment, we group each data column by data type, convert each column one-by-one into each format, and finally aggregate the statistics of the compressed columns. Table 5 and Table 6 show the overall compression performance and statistics over the ∼31k columns in the CodecDB, Public BI and JOB, datasets. We further show the compression ratio CDFs for each data type in Figure 4 where we focus on the effective compression ratio range (0.0, 1.0). Finally, Figure 5 shows, for each data type, CDFs that takes into account the number of distinct values in each column. To avoid
 Integer Double Parquet DICT(-RLE) DICT(-RLE)
Arrow None None ORC RLE None
3.5 Discussion
String/Binary
DICT(-RLE) DICT DICT-RLE
Decimal
DICT(-RLE) None Int-RLE
    Overall, Parquet and ORC provide the most comprehensive com- pression support for common data types, whereas Arrow Feather supports the fewest. ORC provides more auxiliary information for query execution (e.g., its zone map and support for bloom filters). Arrow Feather applies the same compression type to all arrays in the same record batch, whereas Parquet is more granular and allows compression to vary across column chunks. This flexibility enables intelligent encoding and compression selection based on the data features or workload characteristics [33]. As summarized in Table 3, each format applies different default encoding strategies.
In terms of data access, both Arrow and ORC require data to be fully loaded into dedicated in-memory data structures (an Arrow Table or ORC ColumnVectorBatch, respectively) before further query execution can begin. On the other hand, Parquet exposes a streaming API that allows pipelining data parsing and query execu- tion, leading to more optimization opportunities. However, Parquet does not itself provide any dedicated in-memory data structures.
4 METHODOLOGY
In the subsequent sections, we benchmark the performance of Ar- row, Parquet, and ORC over (non-nested) relational data. This is not strictly an apples-to-apples comparison because each format was developed with a different use case in mind: Arrow eases the shar- ing of in-memory data across systems, Parquet is a generic on-disk format, and ORC is a storage format for relational big data systems. Nevertheless, this comparison is important for evaluating the de- sign choices (e.g., encoding method, compression, implementation decisions) made by each format and to understand the limitations and opportunities when using these formats in analytical DBMSs.
Dimensions. To most fairly compare the formats, we evaluate each format across the following dimensions:
1. Compression ratio. Each format applies different encoding methods and supports different compression algorithms. The final achievable compression ratio is a result of these decisions, and so we evaluate each format using the variously supported encoding and compression algorithms (Section 5.1).
2. Transcoding throughput. While compression ratio alone is sufficient if we care only about minimizing disk or memory usage, this comes at the cost of having to compress, convert, and decom- press (i.e., transcode) the data when accessing it (Section 5.2).
3. Data access. For each data type, what are the costs of accessing them? Data is often accessed by column (i.e., projected) or filtered
3047

Table 4: Evaluation overview and key results.
 Evaluation dimension
Compression ratio Compression throughput Decompression throughput Projection evaluation Predicate evaluation Bitmap evaluation
Subexpression evaluation
Direct querying Vectorized execution
Best Overall
Parquet
Arrow Feather Arrow Feather Parquet and ORC ORC
ORC
ORC
Parquet Parquet
Key Advantage Section
Comprehensive encoding and compression support 5.1 Fast serialization 5.2.1 Fast deserialization 5.2.2 Fine-grained skipping while loading data 6.1 Fine-grained loading control with dedicated in-memory representation 6.2 Fine-grained loading control with dedicated in-memory representation 6.2.3 Fine-grained loading control with dedicated in-memory representation
7
and efficient skipping
In-memory mapping with data skipping and direct querying 8 In-memory mapping with data skipping, direct querying, and SIMD support 8
                       Data
Type Integer Float String Total
# Raw Cols. Size 12k 57.3
7k 58.8 13k 373.5 31k 489.7
Parquet ORC Size Size 9.8 13.5
24.0 58.2 31.0 62.2 64.7 133.9 0.13 0.27
Arrow Arrow Size (DICT) 59.3 59.3∗ 59.8 59.8∗
403.4 118.3 522.5 237.4 1.07 0.48
(a) Integers
Figure 4: Column compression ratio CDFs over the CodecDB, BI and JOB datasets. Figure 5: Distinct value CDFs.
Table 5: Total size (in GB) by format for columns in the CodecDB, BI, and JOB datasets. We serialize each column separately into each format and group their compressed size by data type. The raw dataset is in CSV format. For Arrow, we report with dictionary encoding enabled (Arrow DICT) and disabled (the default). We copy the file size (marked with *) from the Arrow default column for CR computation as there is no dictionary support for integer and float types.
a 7% increase in size compared with the raw text file. We found that this overhead is introduced by the format’s metadata, which adds a four-byte length prefix to each variable binary entry (i.e., the string “abc” consumes seven bytes in total). It also pads numerical data types. On the other hand, with DICT enabled, Arrow Feather compresses string columns by 68% and the whole dataset by 52%.
For integers, ORC exhibits varying compression performance relative to Parquet. ORC achieves a better compression ratio for the CodecDB and JOB datasets (which contain a relatively higher number of distinct values), while it is worse for the BI dataset (which has a lower number of distinct values). This is because ORC applies RLE for integer columns (see Table 3), which performs better for columns with fewer distinct values, whereas Parquet applies DICT-RLE, which is slightly worse. Because of this, we observe a crossover point for the Parquet and ORC CDFs in Figure 4a.
For floats, as we can see from Figure 4b, Parquet outperforms ORC and Arrow Feather because of dictionary encoding. ORC and Arrow Feather perform similarly as they both use plain encoding. For strings, Parquet and ORC outperform Arrow. Interestingly, both Parquet and ORC fall back to plain encoding on some columns when dictionary encoding takes up larger space than plain encoding, but their dictionary encoding work differently: Parquet’s plain encoding introduces a higher space cost for saving the string length values, while ORC’s plain encoding uses RLE for string length values. However, Parquet’s dictionary encoding is more effective than ORC because of the extra layer of RLE for the dictionary- encoded keys. That is why Parquet works better in terms of total compressed size (see Tables 5 and 6) while ORC works better in terms of the effectiveness (compression ratio < 1; see Figure 4c).
(b) Floats (c) Strings
      Compression Ratio (CR)
 Table 6: Average and stddev compression ratios by data type.
 Type
Int Float String
Parquet ORC Arrow ArrowDICT AVG STD AVG STD AVG STD AVG STD
  0.25 0.27 0.26 0.34 0.26 1.43 0.21 0.34 0.22
0.18 1.41 1.00 1.49 0.31 1.54
0.84 - - 1.09 - - 0.68 0.92 0.87
   confusion, we do not apply any further compression after default encoding techniques are applied (we will explore how each format behaves when compression is enabled in the following sections).
As we can see in Table 5, overall, Parquet performs the best over the whole dataset and is able to reduce the size of the column data to about 13% of the original. ORC is able to compress the dataset to ∼27%. By contrast, Arrow Feather—with default settings—exhibits
3048

 Compressed Size (MB)
 150 100 50 0
None Zstandard Parquet
LZ4 gzip
Snappy zlib
Arrow Feather ORC
 100% 80% 60% 40% 20% 0%
None
Zstandard LZ4 gzip Compression Codec
Snappy zlib
Parquet
Arrow Feather ORC
  Compression Ratio
Figure 6: Compression ratio (compressed size / original CSV size) on TPC-DS (smaller is better). Uncompressed (None in the figure) only encodes using default settings. Not all formats support all compression algorithms.
Figure 8: Total size on disk after compressing the string columns in TPC-DS.
integer (both int32 and int64), double, and string data types. We extract all columns of a given data type, compress them, and report the aggregate sizes by type. The results are in Figures 7 and 8.
First, consider Figure 7a which shows the aggregate compres- sion performance on the integer columns. ORC achieves slightly better compression performance than Parquet. This is because Par- quet applies DICT and switches to plain encoding for some of the columns, whereas ORC always applies RLE. Arrow Feather does not encode by default. This leads to the worst compression ratio when compression is disabled. Nevertheless, all three data formats perform similarly when compression is enabled, except for LZ4, where Arrow Feather is almost 50% worse because it lacks encoding support for integers (we observe a similar result in Figure 6).
Next, Figure 7b shows the aggregate compression performance for the double columns. Parquet also applies DICT to this data type, whereas ORC and Arrow Feather do not encode at all. Because of this, Arrow and ORC have very similar performance both in the uncompressed and compressed setting, whereas Parquet is slightly better. The ORC outlier for LZ4 happens for the same reason as discussed in Section 5.1.2.
Finally, Figure 8 shows compression performance on string columns (both variable- and fixed-length). By default, Arrow does not encode this type, whereas ORC and Parquet apply DICT. Among all formats, Parquet has the best compression performance, followed by ORC and Arrow. ORC produces larger compressed sizes than Parquet, because: (i) ORC has a smaller default block size and thus pays more dictionary overhead per row batch; and (ii) it more frequently falls back to plain encoding because of its row batch-level dictionary encoding (versus the chunk-level used in Parquet). Again, LZ4 ORC disables compression because it offers no benefit.
5.2 Transcoding Overhead
In practice, storage formats are converted into (or from) an in- memory presentation on reads (writes). We now evaluate the over- heads in transcoding (i.e., decompressing, converting, and com- pressing) each format. Specifically, Section 5.2.1 explores the time required to compress and serialize each format from a common in- memory representation, while Section 5.2.2 evaluates the overhead of loading data, i.e., deserializing and decompressing each format into an in-memory representation amenable to query execution.
5.2.1 Compression Overhead. Our first experiment in this sec- tion explores how long it takes to serialize (and compress) the data from an in-memory representation to each disk format. For this experiment we use the catalog_sales TPC-DS table. The catalog_sales is a large (∼14M rows) and wide (34 columns) ta- ble containing integers and doubles. Its raw data size is 3GB. All
6 4 2 0
6 4 2 0
None
None
Zstandard
LZ4 gzip
Snappy zlib
P(a)rqInuetetgersA(rirnotw64Feaanthdeirnt32O)RC
Zstandard Parquet
LZ4 gzip
Snappy zlib
Arrow Feather ORC (b) Doubles
Figure 7: Total size on disk after compressing the numeric columns in TPC-DS.
5.1.2 Compression Performance. For this experiment, we report the compression ratio of each format on the full TPC-DS dataset when different compression algorithms are applied. We evaluate Zstandard (Zstd) at level 1 (we evaluate other levels later in this section), LZ4, Gzip, Snappy, and Zlib compression algorithms, and compare them against an uncompressed variant where data is en- coded using the default settings. The results of this experiment are shown in Figure 6. In the uncompressed case, Parquet is about 2× better than Arrow Feather because Arrow Feather does not apply any encoding. However, when compression is enabled, Arrow per- forms within ∼30% of Parquet. ORC achieves a similar compression ratio as Parquet, except under LZ4. In this case, ORC automatically disables compression because it detects that the LZ4 compressed data size is greater than the original data size.
Finally, we observe that different compression algorithms yield different compression ratios. For example, increasing Zstd’s level from 5 to 9 yields more aggressive compression and achieves smaller sizes. However, this gain is minimal (< 1.5%) while the compression time increases by ∼3× for Arrow Feather and ∼2× for Parquet. We will show in Section 5.2 how decompression costs are also impacted by the choice of compression algorithm.
5.1.3 Compression Performance by Data Type. In this experiment, we look at the performance over various column types in the TPC- DS dataset. Specifically, we evaluate compression performance on
3049
  re
gend
Le
moved
Figure 3(b) Figure 3(a) Figure 2 Compressed Size (GB) Compressed Size (GB)
Figure 4(a)

        ~
360 245 30 115 00
3 2 1
0.02 0 NoNnoene ZsZtsdtd LZL4Z4 ggzzipip SSnappy zlib
(a) Writing to disk
Figure 9: Write time from an Arrow in-memory table to each format stored either on disk or in memory.
Figure 10: Runtime (in seconds) for decompressing the TPC- DS catalog_sales table from the on-disk formats into in- memory Arrow.
formats support serializing from an Arrow Table, and so we adopt it as our common in-memory representation.
Figure 9 shows the runtimes when: (i) writing to disk (9a); (ii) writing to the null device, which avoids any I/O overhead (9b); as well as (iii) the data sizes per format (9c). We omit the LZ4 and Snappy bars for ORC as the Apache Arrow C++ library has limited compression support for the ORC format. Starting with Figure 9a, we can see that Arrow Feather is the most efficient format in terms of compression and serialization runtime because it does not encode data. On the other hand, Arrow Feather’s lack of encoding leads to almost a 50% larger footprint on disk (Figure 9c). Interestingly, ORC compression time is 50% slower than Parquet with comparable or slightly better compression ratio on disk (up to 15% better). We think that this is because of better Parquet support in Arrow; both projects share the same codebase and data structures.
Finally, we isolate the compression overhead in Figure 9b by avoiding disk I/O by writing to the null device. Here we can see a decrease in runtime for all the formats, although of different magnitudes. Arrow Feather has the biggest difference, thanks to its inherent zero-copy implementation in Arrow. The compression time for Parquet and ORC does not change substantially because the encoding and compression operations dominate the total runtime.
5.2.2 Decompression Overhead (i.e., table scan). In this experiment we investigate the overhead of loading the catalog_sales TPC- DS table from disk into memory. Our goal with this experiment is to simulate the overheads involved when a query processor is required to load and transform a compressed dataset into a plain in-memory format amenable to query execution. We start from data on disk in the Parquet, ORC, or Arrow Feather formats, and we report the time required to load the data and convert it into the Arrow in-memory format. The results are shown in Figure 10.
Interestingly, loading compressed data has 30% less overhead than the uncompressed case for Arrow under LZ4. This is because LZ4 requires less disk I/O (since the file on disk is smaller; see Figure 9c) while also providing “fast enough” decompression rela- tive to the other compression methods. For the other cases, Arrow
Figure 11: Runtime (in seconds) for decompressing the TPC-DS catalog_sales table from the formats in memory (ramdisk) into in-memory Arrow.
always exhibits the best performance: this is expected since it does not require decoding the data and its on-disk compressed size is reasonable. Parquet is slightly worse than Arrow because of the cost of decoding data, while ORC has the worst performance (it is particularly bad for Zstd and zlib). We think that this is due to de- compression settings such as block size, buffer size, etc. In general, for formats that heavily leverage encoding (i.e., Parquet and ORC) data compression leads to a heavy penalty on read performance.
To isolate disk I/O from compression overheads, we load each compressed dataset onto a memory-resident disk mounted on tmpfs. As we can see from Figure 11, in all cases the runtimes decrease, especially for Arrow without compression. This result is intuitive because, for uncompressed data, the data size is much larger and disk bandwidth is saturated. Conversely, decompression is CPU- bound and not substantially impacted by the cost of bringing data into memory. Combined with previous compression experiments in Figure 9, this shows the benefits of Arrow as a fast inter-process format, when disk I/O and size are not the bottleneck.
To summarize, encoding and compression choices greatly impact performance, with formats like Parquet and ORC targeting size on disk, while Arrow targets raw read performance. To optimize both size and performance, formats should be carefully tuned to the workload and use case, and workload-aware compression selection is crucial. It remains an open question of how much computation can be pushed into the encoded space to minimize the decoding step while maximizing the compression ratio.
6 DATA ACCESS MICROBENCHMARKS
Having explored the overheads associated with encoding, com- pression, and scan operations, we next evaluate the performance of accessing data in the context of common relational operations found near the leaves of a query plan. Specifically, we explore the performance of projecting columns in a dataset (Section 6.1) and applying filters (Section 6.2). In this and subsequent sections we only consider Zstd and LZ4 compression since we evaluated the trade-offs of the other compression algorithms in Section 5.
(b) Writing to null device
(c) Compressed size on disk
3050
None Zstd LZ4 gzip Snappy zlib
 60 45 30 15
0
None Zstd LZ4 gzip Snappy zlib
Write Time (sec) Write Time
 169
Figu
 Parquet Arrow Feather ORC
  ~
167
Fig
 15 10 5 0
None Zstandard Parquet
LZ4 gzip
Snappy zlib
Arrow Feather ORC
  Runtime (sec)
 15 10 5 0
None Zstandard Parquet
LZ4 gzip
Snappy zlib
Arrow Feather ORC
 Runtime (sec)
Figure 6
Figure 5(a)
Figure 7
Figure 5 legend re 5(b)
Now please answer the questions: What is contribution of the paper?
    Please only answer the question I asked above and print the answer in a single line.'''
messages = [{"role": "user", "content": input_text}]
input_ids = tokenizer.apply_chat_template(messages, return_dict=True, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))

# input_text = "Databricks was founded in "
# input_ids = tokenizer(input_text, return_tensors="pt")
#
# outputs = model.generate(**input_ids, max_new_tokens=100)
# print(tokenizer.decode(outputs[0]))