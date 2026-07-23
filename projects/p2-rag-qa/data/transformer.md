# Transformer

Transformer 使用注意力机制建模 token 之间的关系。每个 token 经过线性映射得到 Query、Key、Value；Query 与 Key 的相似度决定从各个 Value 聚合多少信息。

Encoder 可以同时看到输入左右两侧，适合语言理解。Decoder 使用因果掩码，只能看到当前位置之前的 token，适合自回归生成。Encoder-Decoder 架构常用于翻译和摘要。

Embedding 把离散文本映射为连续向量。语义相近文本的向量方向通常更接近，因此可用余弦相似度做语义检索。
