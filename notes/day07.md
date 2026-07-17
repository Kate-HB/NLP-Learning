# 学习曲线
学习曲线是模型在训练过程中性能指标随时间变化的直观表示。需要重点关注的两条曲线是：

损失曲线：显示模型误差（损失）如何随训练步骤或轮数变化
准确率曲线：显示训练步骤或轮数内正确预测的百分比
这些曲线有助于我们了解模型的学习效率，并指导我们进行调整以提升性能。在 Transformer 模型中，这些指标会针对每个批次单独计算，然后记录到磁盘。之后，我们可以使用Weights & Biases等库来可视化这些曲线，并跟踪模型性能随时间的变化。

## 损失曲线[[loss-curves]]
损失曲线显示了模型误差随时间推移的减小情况。在一次典型的成功训练过程中，你会看到类似下图的曲线：
[图示](03-NLP-Basic\pictures\损失曲线.png)
初始损失高：模型未经优化就启动，因此初始预测效果较差。
损失减少：随着训练的进行，损失通常会减少。
收敛：最终，损失值稳定在一个较低的水平，表明模型已经学习到了数据中的模式。

## 准确率曲线[[accuracy-curves]]
准确率曲线显示了预测正确率随时间的变化。与损失曲线不同，准确率曲线通常会随着模型学习的进行而增加，并且通常可以包含比损失曲线更多的步骤。

初始准确率应较低：由于模型尚未学习数据中的模式，因此初始准确率应该较低。
随着训练的进行，准确率会提高：如果模型能够学习数据中的模式，那么随着模型的学习，准确率通常也会提高。
可能出现平台期：准确率的提升通常呈离散跳跃式而非平滑式，因为模型会做出接近真实标签的预测。

为什么准确率曲线呈“阶梯状”：与连续的损失函数不同，准确率是通过将离散的预测结果与真实标签进行比较来计算的。模型置信度的微小提升可能不会改变最终的预测结果，导致准确率在达到某个阈值之前保持平稳。

## 收敛[[convergence]]
当模型性能趋于稳定，损失曲线和准确率曲线趋于平缓时，模型就收敛了。这表明模型已经学习了数据中的模式，可以投入使用了。简单来说，我们的目标是让模型每次训练后都能收敛到稳定的性能水平。
[图示](03-NLP-Basic\pictures\收敛.png)
一旦模型收敛，我们就可以使用它们对新数据进行预测，并参考评估指标来了解模型的性能如何。

## 健康的学习曲线
一次表现良好的训练运行通常会呈现出类似下图的曲线形状：
[图示](03-NLP-Basic\pictures\收敛.png)
图中左侧是损失曲线，右侧是对应的准确率曲线。这两条曲线具有截然不同的特征。

损失曲线显示了模型损失值随时间的变化。初始阶段损失值较高，然后逐渐下降，表明模型性能正在提升。损失值的下降表明模型预测效果越来越好，因为损失值代表预测输出与真实输出之间的误差。

准确率曲线上代表了模型准确率随时间的变化。准确率曲线初始值较低，随着训练的进行而逐渐升高。准确率衡量的是正确分类的实例比例。因此，准确率曲线越高，表明模型的预测越准确。

两条曲线的一个显著区别在于准确率曲线的平滑度和“平台”的存在。损失曲线平滑下降，而准确率曲线上的平台则表明准确率出现了离散的跃升，而非持续上升。这种现象归因于准确率的衡量方式。即使最终预测仍然错误，只要模型的输出更接近目标值，损失值也会有所改善。然而，只有当预测结果超过正确阈值时，准确率才会提高。

例如，在一个区分猫（0）和狗（1）的二元分类器中，如果模型预测一张狗的图像（真实值为1）的值为0.3，则该值会被四舍五入为0，因此分类错误。如果在下一步中模型预测值为0.4，则分类仍然错误。损失值会降低，因为0.4比0.3更接近1，但准确率保持不变，形成一个平台期。只有当模型预测值大于0.5且四舍五入为1时，准确率才会显著提升。

健康曲线的特征：
损失平稳下降：训练损失和验证损失均稳步下降。
训练/验证性能接近：训练指标和验证指标之间的差距很小
收敛：曲线趋于平缓，表明模型已学习到这些模式。


## 过拟合
当模型从训练数据中学习到太多东西，而无法泛化到不同的数据（由验证集表示）时，就会发生过拟合。
[图示](03-NLP-Basic\pictures\过拟合.png)
症状：
训练损失持续下降，而验证损失则上升或趋于稳定。
训练准确率和验证准确率之间存在较大差距
训练准确率远高于验证准确率。

解决过拟合问题的方案：
正则化：添加 dropout、权重衰减或其他正则化技术
提前停止：当验证性能不再提升时停止训练。
数据增强：增加训练数据的多样性
降低模型复杂度：使用更小的模型或更少的参数。

## 欠拟合
当模型过于简单，无法捕捉数据中的潜在模式时，就会发生欠拟合。
[图示](03-NLP-Basic\pictures\欠拟合.png)
这种情况可能由以下几个原因造成：
该模型太小或缺乏学习模式的能力
学习率太低，导致学习速度缓慢
数据集太小或不具有代表性。
该模型未进行适当的正则化。

症状：
训练损失和验证损失均较高。
模型性能在训练初期就趋于平稳
训练准确率低于预期

解决欠拟合的方法：
增加模型容量：使用更大的模型或更多参数
延长训练时间：增加训练轮数
调整学习率：尝试不同的学习率
检查数据质量：确保您的数据已正确预处理。


## 不稳定的学习曲线
当模型学习效率低下时，就会出现不稳定的学习曲线。
[图示](03-NLP-Basic\pictures\不稳定1.png)
这可能是由以下几个原因造成的：
学习率过高，导致模型过度调整参数。
批次大小太小，导致模型学习速度慢。
该模型没有进行适当的正则化，导致其过拟合训练数据。
数据集未经过适当预处理，导致模型从噪声中学习。

症状：
损失或准确性频繁波动
曲线显示出较高的方差或不稳定性
业绩波动不定，没有明显趋势。
训练曲线和验证曲线均表现出不稳定的行为。
[图示](03-NLP-Basic\pictures\不稳定2.png)

解决不规则曲线问题：
降低学习率：减小步长以获得更稳定的训练效果
增加批次大小：更大的批次可以提供更稳定的梯度。
渐变裁剪：防止渐变爆炸
更佳的数据预处理：确保数据质量的一致性


# 使用hugging face的预训练模型
假设我们正在寻找一种可以执行掩码填充（mask filling 又称完形填空）的 French-based（法语）模型。

from transformers import pipeline
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
[
  {'sequence': 'Le camembert est délicieux :)', 'score': 0.49091005325317383, 'token': 7200, 'token_str': 'délicieux'}, 
]
在管道中加载模型非常简单。唯一需要注意的是所选 checkpoint 是否适合它将用于的任务。
例如，这里我们正在将 camembert-base checkpoint 加载在 fill-mask 管道，这完全没问题。
但是如果我们在 text-classification 管道中加载该 checkpoint 结果没有任何意义，因为 camembert-base 不适合这个任务！
使用预训练模型时，一定要检查它是如何训练的、在哪些数据集上训练的、它的局限性和偏见。所有这些信息都应在其模型卡片上有所展示。

# 共享预训练模型
创建模型存储仓库的方法有以下三种：
使用 push_to_hub API 接口
使用 huggingface_hub Python 库
使用网页界面
创建仓库后，你可以通过 git 和 git-lfs 将文件上传到其中。

## 使用 push_to_hub API

生成一个身份验证令牌，这样 huggingface_hub API 才会知道你是谁以及你对哪些空间具有写入权限。

运行以下代码登录：
from huggingface_hub import notebook_login
notebook_login()

在终端中，你可以运行：
huggingface-cli login


使用 Trainer API 训练模型，将其上传到 Hub 的最简单方法是在定义 TrainingArguments 时设置 push_to_hub=True 

from transformers import TrainingArguments
training_args = TrainingArguments(
    "bert-finetuned-mrpc", save_strategy="epoch", push_to_hub=True
)
这样，调用 trainer.train() 的时候， Trainer 会在每次保存模型时（这里是每个训练周期）将其上传到 Hub 中你的账户下的一个仓库。该仓库的名称与你选择的输出目录名称相同（这里是 bert-finetuned-mrpc ），但你可以通过 hub_model_id = "a_different_name" 指定一个不同的名称。

若要将你的模型上传到你所属的组织，只需设置 hub_model_id = my_organization/my_repo_name 。

最后，需要在训练循环结束后运行 trainer.push_to_hub() 上传模型的最新版本。它还会生成一份包含所有相关元数据的模型卡片，包含使用的超参数和评估结果！
以下是一个模型卡片的示例：
[图示](03-NLP-Basic\pictures\model_card.png)


# DATASETS库
数据集不在 Hub 

## 使用本地和远程数据集
Datasets 提供了加载本地和远程数据集的方法。它支持几种常见的数据格式，例如：

数据格式	        类型参数	        加载的指令
CSV & TSV	        csv	            load_dataset("csv", data_files="my_file.csv")
Text files	        text	        load_dataset("text", data_files="my_file.txt")
JSON & JSON Lines	json	        load_dataset("json", data_files="my_file.jsonl")
Pickled DataFrames	pandas	        load_dataset("pandas", data_files="my_dataframe.pkl")
如表所示，对于每种数据格式，我们只需要在 load_dataset() 函数中指定数据的类型，并使用 data_files 指定一个或多个文件的路径的参数。

### 加载本地数据集
在这个例子中，我们将使用 SQuAD-it 数据集 ，这是一个用于意大利语问答的大规模数据集。
训练集和测试集都托管在 GitHub 上，因此我们可以通过 wget 命令非常轻易地下载它们：

curl -o SQuAD_it-train.json.gz https://github.com/crux82/squad-it/raw/master/SQuAD_it-train.json.gz
curl -o SQuAD_it-test.json.gz https://github.com/crux82/squad-it/raw/master/SQuAD_it-test.json.gz
这将下载两个名为 SQuAD_it-train.json.gz 和 SQuAD_it-test.json.gz 的压缩文件，在git bash解压他们：
gzip -dkv SQuAD_it-*.json.gz
<!-- 注意: Windows PowerShell 中 wget 是 Invoke-WebRequest 的别名，不会保存文件，应用 curl -o 替代。gzip 在 PowerShell 不可用，需在 Git Bash 中执行。 -->
我们可以看到压缩文件已经被替换为 SQuAD_it-train.json 和 SQuAD_it-test.json ，并且数据以 JSON 格式存储。

当我们使用 load_dataset() 函数来加载 JSON 文件时，我们需要知道我们是在处理普通的 JSON（类似于嵌套字典）还是 JSON Lines（每一行都是一个 JSON）。
JSON==JavaScript Object Notation。一种纯文本数据格式，键值对结构，人和机器都能直接读。
像许多问答数据集一样，SQuAD-it 使用的是嵌套字典，所有文本都存储在 data 字段中。这意味着我们可以通过使用参数 field 来加载数据集，如下所示：
from datasets import load_dataset
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")

默认情况下，加载本地文件会创建一个带有 train 标签的 DatasetDict 对象。我们可以在这里查看一下 squad_it_dataset 对象：
squad_it_dataset
输出了训练集的行数和列名。我们可以使用 train 标签来查看数据集中的一个示例，如下所示：
squad_it_dataset["train"][0]

仅仅加载了训练集，我们真正想要的是包含 train 和 test 的 DatasetDict 对象。这样的话就可以使用 Dataset.map() 函数同时处理训练集和测试集。为此，我们向 data_files 参数输入一个字典，将数据集的标签名映射到相关联的文件：
data_files = {"train": "SQuAD_it-train.json", "test": "SQuAD_it-test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
squad_it_dataset
这正是我们想要的。现在，我们可以使用各种预处理技术来清洗数据、tokenize 评论等等。

load_dataset() 函数的 data_files 参数非常灵活：可以是单个文件路径、文件路径列表或者是标签映射到文件路径的字典。

Datasets 实际上支持自动解压输入文件，所以我们可以跳过使用 gzip ，直接将 data_files 参数设置为压缩文件：
data_files = {"train": "SQuAD_it-train.json.gz", "test": "SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

### 加载远程数据集
加载远程文件只需要将 load_dataset() 的 data_files 参数指向存储远程文件的一个或多个 URL。例如，对于托管在 GitHub 上的 SQuAD-it 数据集，我们可以将 data_files 设置为指向 SQuAD_it-*.json.gz 的网址，如下所示：

url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")

这将返回和上面的本地例子相同的 DatasetDict 对象，但省去了我们手动下载和解压 SQuAD_it-*.json.gz 文件的步骤。




## 分割和整理数据
### 分割和整理我们的数据
与 Pandas 类似，Datasets 提供了多个函数来操作 Dataset 和 DatasetDict 对象。如Dataset.map() 方法，在本节中，我们将探索一些其他可用的函数。

在本例中，我们将使用托管在 加州大学欧文分校机器学习仓库 的 药物审查数据集 ，其中包含患者对各种药物的评论，以及正在治疗的病情和患者满意度的 10 星评价。

首先我们需要下载并解压数据，可以通过 wget 和 unzip 命令：

# 注意: UCI 已将数据集迁移，原 URL https://archive.ics.uci.edu/ml/machine-learning-databases/00462/ 已失效。
# 备选方案1—通过HuggingFace加载:
#   load_dataset("csv", data_files={"train": "https://huggingface.co/datasets/kyleiwaniec/drugscom-review/resolve/main/drugsComTrain_raw.tsv", "test": "..."}, delimiter="\t")
# 备选方案2—使用 ucimlrepo 包:
#   pip install ucimlrepo
#   from ucimlrepo import fetch_ucirepo; fetch_ucirepo(id=462)
!unzip drugsCom_raw.zip
由于 TSV 仅仅是 CSV 的一个变体，它使用制表符而不是逗号作为分隔符，我们可以使用加载 csv 文件的 load_dataset() 函数并指定分隔符，来加载这些文件：
from datasets import load_dataset
data_files = {"train": "drugsComTrain_raw.tsv", "test": "drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

在进行数据分析时，获取一个小的随机样本以快速了解你正在处理数据的特点是一种好的实践。在数据集中，我们可以通过链接 Dataset.shuffle() 和 Dataset.select() 函数创建一个随机的样本：
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
drug_sample[:3]

请注意，出于可以复现的目的，我们已将在 Dataset.shuffle() 设定了固定的随机数种子。 Dataset.select() 需要一个可迭代的索引，所以我们传递了 range(1000) 从随机打乱的数据集中抽取前 1,000 个示例。从抽取的数据中，我们已经可以看到我们数据集中有一些特殊的地方：
Unnamed: 0 这列看起来很像每个患者的匿名 ID。
condition 列包含了大小写混合的标签。
评论长短不一，混合有 Python 行分隔符 （ \r\n ） 以及 HTML 字符代码，如 &\#039; 。

如何使用Datasets 来处理这些问题。为了验证 Unnamed: 0 列存储的是患者 ID 的猜想，我们可以使用 Dataset.unique() 函数来验证匿名 ID 的数量是否与分割后每个分组中的行数匹配：
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

这似乎证实了我们的假设，所以让我们把 Unnamed: 0 列重命名为患者的 id。我们可以使用 DatasetDict.rename_column() 函数来一次性重命名两个分组：

drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)


接下来，让我们使用 Dataset.map() 来规范所有的 condition 标签。正如我们在 第三章 中处理 tokenizer 一样，我们可以定义一个简单的函数，可以使用该函数 drug_dataset 处理每个分组的所有行：
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}
drug_dataset.map(lowercase_condition)
=>
AttributeError: 'NoneType' object has no attribute 'lower'
哦不，我们的 map 函数遇到了问题！从错误中我们可以推断出 condition 列存在 None ，不能转换为小写，因为它们不是字符串。让我们使用 Dataset.filter() 删除这些行 其工作方式类似于 Dataset.map() 。例如：
def filter_nones(x):
    return x["condition"] is not None

然后运行 drug_dataset.filter(filter_nones) ，我们可以用 lambda 函数在一行代码完成这个任务。在 Pyhton 中，lambda 函数是你无需明确命名即可使用的微函数（匿名函数）。它们一般采用如下形式：
lambda <arguments> : <expression>
其中 lambda 是 Python 的特殊 关键字 之一， arguments 是以逗号进行分隔的函数参数的列表/集合， expression 代表你希望执行的操作。例如，我们可以定义一个简单的 lambda 函数来对一个数字进行平方，如下所示：
lambda x : x * x
我们需要将要输入放在括号中：
(lambda x: x * x)(3)
Copied
9
同样，我们可以通过使用逗号分隔来定义带有多个参数的 lambda 函数。例如，我们可以按如下方式计算三角形的面积：
(lambda base, height: 0.5 * base * height)(4, 8)
Copied
16.0

在Datasets 中，我们可以使用 lambda 函数来定义简单的映射和过滤操作，所以让我们使用这个技巧来删除我们数据集中的所有condition 为 None的记录：
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

含有None的积累删除之后,我们可以规范我们的 condition 列:
drug_dataset = drug_dataset.map(lowercase_condition)

检查一下转换后的结果
drug_dataset["train"]["condition"][:3]
['left ventricular dysfunction', 'adhd', 'birth control']
有用！现在我们已经清理了标签，让我们来看看清洗后的评论文本。

### 创建新的列
每当我们处理客户评论时，一个好的习惯是检查评论的字数的分布。评论可能只是一个词，比如“太棒了！”或包含数千字的完整文章。在不同的使用场景，你需要以不同的方式处理这些极端情况。为了计算每条评论中的单词数，我们将使用空格分割每个文本进行粗略统计。

定义一个简单的函数，计算每条评论的字数：
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
不同于我们的 lowercase_condition() 函数， compute_review_length() 返回一个字典，其键并不对应数据集中的某一列名称。在这种情况下，当 compute_review_length() 传递给 Dataset.map() 时，它将处理数据集中的所有行，最后返回值会创建一个新的 review_length 列：
drug_dataset = drug_dataset.map(compute_review_length)

检查第一个训练样例
drug_dataset["train"][0]

{'patient_id': 206461,
 'drugName': 'Valsartan',
 'condition': 'left ventricular dysfunction',
 'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"',
 'rating': 9.0,
 'date': 'May 20, 2012',
 'usefulCount': 27,
 'review_length': 17}
正如预期的那样，我们可以看到一个 review_length 列已添加到我们的训练集中。我们可以使用 Dataset.sort() 对这个新列进行排序，然后查看一下极端长度的评论是什么样的：
drug_dataset["train"].sort("review_length")[:3]

{'patient_id': [103488, 23627, 20558],
 'drugName': ['Loestrin 21 1 / 20', 'Chlorzoxazone', 'Nucynta'],
 'condition': ['birth control', 'muscle spasm', 'pain'],
 'review': ['"Excellent."', '"useless"', '"ok"'],
 'rating': [10.0, 1.0, 6.0],
 'date': ['November 4, 2008', 'March 24, 2017', 'August 20, 2016'],
 'usefulCount': [5, 2, 10],
 'review_length': [1, 1, 1]}
正如我们所猜想的那样，有些评论只包含一个词，虽然这对于情感分析任务来说还可以接受，但如果我们想要预测病情，那么它所提供的信息就不够丰富了。

向数据集添加新列的另一种方法是使用函数 Dataset.add_column() ，在使用它时你可以通过 Python 列表或 NumPy 数组的方式提供数据，在不适合使用 Dataset.map() 情况下可以很方便。

使用 Dataset.filter() 功能来删除包含少于 30 个单词的评论。这与我们过滤 condition 列的处理方式相似，我们可以通过设定评论长度的最小阈值，筛选出过短的评论：
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)

我们需要处理的最后一件事是处理评论中的 HTML 字符。我们可以使用 Python 的 html 模块来解码这些字符，如下所示：
import html
text = "I&#039;m a transformer called BERT"
html.unescape(text)
=>
"I'm a transformer called BERT"

我们将使用 Dataset.map() 对我们语料库中的所有 HTML 字符进行解码：
drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

### map() 方法的超级加速
Dataset.map() 方法有一个 batched 参数，如果设置为 True ，map 函数将会分批执行所需要进行的操作（批量大小是可配置的，但默认为 1,000）。例如，之前对所有 HTML 进行解码的 map 函数运行需要一些时间（你可以从进度条中看到所需的时间）。我们可以通过使用列表推导同时处理多个元素来加速。

当你在使用 Dataset.map() 函数时设定 batched=True 。该函数需要接收一个包含数据集字段的字典，字典的值是一个列表。例如，这是使用 batched=True 对所有 HTML 字符进行解码的方法
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)

此命令的执行速度比前一个命令快得多。这是因为列表推导式通常比在同一代码中用 for 循环执行相同的代码更快，并且我们还通过同时访问多个元素而不是一个一个来处理来提高处理的速度。

在第六章我们将遇到的“快速” tokenizer 它可以快速对长文本列表进行 tokenize。使用 Dataset.map() 搭配 batched=True 参数是加速的关键。例如，要使用快速 tokenizer 对所有药物评论 tokenize，我们可以使用如下的函数：
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)

正如我们在 第三章 所看到的，我们原本就可以将一个或多个示例传递给 tokenizer，因此在 batched=True 是一个非必须的选项。让我们借此机会比较不同选项的性能。在 notebook 中，你可以在你要测量的代码行之前添加 %time 来记录该行运行所消耗的时间：
%time tokenized_dataset = drug_dataset.map(tokenize_function, batched=True)
你也可以将 %%time 放置在单元格开头来统计整个单元格的执行时间。

在有和无 batched=True 的情况下执行相同的指令，然后试试慢速 tokenizer （在 AutoTokenizer.from_pretrained() 方法中添加 use_fast=False ），这样你就可以测试一下在你的电脑上它需要多长的时间。

使用快速 tokenizer 配合 batched=True 选项比没有批处理的慢速版本快 30 倍！这就是为什么在使用 AutoTokenizer 时，将会默认使用 use_fast=True 的主要原因。他们能够实现这样的加速，因为在底层的 tokenize 代码是在 Rust 中执行的，Rust 是一种可以易于并行化执行的语言。

并行化也是快速 tokenizer 通过批处理实现近 6 倍加速的原因：单个 tokenize 操作是不能并行的，但是当你想同时对大量文本进行 tokenize 时，你可以将执行过程拆分为多个进程，每个进程负责处理自己的文本。 Dataset.map() 也有一些自己的并行化能力。尽管它们没有 Rust 提供支持，但它们仍然可以帮助慢速 tokenizer 加速（尤其是当你使用的 tokenizer 没有快速版本时）。要启用多进程处理，请在调用 Dataset.map() 时使用 num_proc 参数并指定要在调用中使用的进程数
slow_tokenizer = AutoTokenizer.from_pretrained("bert-base-cased", use_fast=False)

def slow_tokenize_function(examples):
    return slow_tokenizer(examples["review"], truncation=True)

tokenized_dataset = drug_dataset.map(slow_tokenize_function, batched=True, num_proc=8)

对于 num_proc 的其他值，在我们的测试中，使用 batched=True 而不带有 num_proc 参数的选项处理起来更快。总的来说，我们并不推荐在快速 tokenizer 和 batched=True 的情况下使用 Python 的多进程处理。

通常来说，使用 num_proc 以加快处理速度通常是一个好主意，只要你使用的函数本身没有进行某种类型的多进程处理。

将所有这些功能浓缩到一个方法中已经非常了不起，但是还有更多！使用 Dataset.map() 和 batched=True 你可以更改数据集中的元素数量。当你想从一个样本中创建几个训练特征时，这是非常有用的。我们将在 第七章 中几个 NLP 任务的预处理中使用到这个功能，它非常便捷。

在机器学习中，一个样本通常可以为我们的模型提供一组特征。在某些情况下，这组特征会储存在数据集的几个列，但在某些情况下（例如此处的例子和用于问答的数据），可以从单个样本的那一列中提取多个特征。

从一列中提取多个特征是如何实现的！在这里，我们将对我们的样本进行 tokenize 并将最大截断长度设置为 128，并且我们将要求 tokenizer 返回全部文本块，而不仅仅是第一个。这可以通过设置 return_overflowing_tokens=True 来实现：

def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
在使用 Dataset.map() 正式开始处理整个数据集之前，让我们先在一个样本上测试一下：
result = tokenize_and_split(drug_dataset["train"][0])
[len(inp) for inp in result["input_ids"]]

[128, 49]
瞧！我们在训练集中的第一个样本变成了两个特征，因为它超过了我们指定的最大截断长度，因此被截成了两段：第一段长度为 128 第二段长度为 49 现在让我们对数据集的所有样本执行此操作！
tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)

ArrowInvalid: Column 1 named condition expected length 1463 but got length 1000
不好了！这并没有成功！为什么呢？查看错误消息会给我们一个线索：列的长度不匹配，一列长度为 1,463，另一列长度为 1,000。1,000 行的“重新”生成了 1,463 行的新特征，导致和原本的 1000 行的长度不匹配。

问题出在我们试图混合两个长度不同的数据集： drug_dataset 列将有 1000 个样本，但是我们正在构建 tokenized_dataset 列将有 1,463 个样本（因为我们使用 return_overflowing_tokens=True 将长评论分词成了多个样本）。这对 Dataset 来说不可行，所以我们需要要么删除旧数据集的列，要么使它们与新数据集中的尺寸相同。我们可以使用 remove_columns 参数来实现前者：
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
现在这个过程没有错误。我们可以通过比较长度来检查我们的新数据集是否比原始数据集有更多的元素：
len(tokenized_dataset["train"]), len(drug_dataset["train"])
(206772, 138514)
我们也可以通过使旧列与新列保持相同大小来处理不匹配长度的问题。为此，当我们设置 return_overflowing_tokens=True 时，可以使用 overflow_to_sample_mapping 字段。它给出了新特征索引到它源自的样本索引的映射。使用这个，我们可以将原始数据集中的每个键关联到一个合适大小的值列表中，通过遍历所有的数据来生成新特性：

def tokenize_and_split(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # 提取新旧索引之间的映射
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result
可以看到它可以与 Dataset.map() 一起协作，无需我们删除旧列：

tokenized_dataset = drug_dataset.map(tokenize_and_split, batched=True)
tokenized_dataset

DatasetDict({
    train: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 206772
    })
    test: Dataset({
        features: ['attention_mask', 'condition', 'date', 'drugName', 'input_ids', 'patient_id', 'rating', 'review', 'review_length', 'token_type_ids', 'usefulCount'],
        num_rows: 68876
    })
})
我们获得了与之前数量相同的训练特征，并且在这里我们保留了所有旧字段。如果你在使用模型计算之后需要它们进行一些后续处理，你可能需要使用这种方法。

###  Datasets 和 DataFrames 的相互转换

为了实现各种第三方库之间的转换，Datasets 提供了一个 Dataset.set_format() 函数。此函数可以通过仅更改输出格式的，轻松切换到另一种格式，而不会影响底层数据格式（以 Apache Arrow 方式进行存储）。为了演示，让我们把数据集转换为 Pandas：
drug_dataset.set_format("pandas")

现在，当我们访问数据集的元素时，我们会得到一个 pandas.DataFrame 而不是字典：
drug_dataset["train"][:3]

接下来我们从数据集中选择 drug_dataset[train] 的所有数据来得到训练集数据：
train_df = drug_dataset["train"][:]
Dataset.set_format() 仅仅改变了数据集的 __getitem__() 方法的返回格式。这意味着当我们想从 "pandas" 格式的 Dataset 中创建像 train_df 这样的新对象时，我们需要对整个数据集进行切片（[:]）才可以获得 pandas.DataFrame 对象。无论输出格式如何，你都可以自己验证 drug_dataset["train"] 的类型依然还是 Dataset 。

链式操作计算 condition 列中不同类别的分布
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)
frequencies.head()

当我们完成了 Pandas 分析之后，我们可以使用对象 Dataset.from_pandas() 方法可以创建一个新的 Dataset 对象，如下所示：
from datasets import Dataset
freq_dataset = Dataset.from_pandas(frequencies)


### 创建验证集
尽管我们有一个可以用于评估的测试集，但在开发过程中保持测试集不变并创建一个单独的验证集是一个很好的做法。一旦你对模型在测试集上的表现感到满意，你就可以使用验证集进行最终的检查。此过程有助于降低你过拟合测试集和部署在现实世界数据上失败的模型的风险。

Datasets 提供了一个基于 scikit-learn 的经典方法： Dataset.train_test_split() 。让我们用它把我们的训练集分成 train 和 validation 
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
将默认的 "test" 部分重命名为 "validation"
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")

将 "test" 部分添加到我们的 `DatasetDict` 中
drug_dataset_clean["test"] = drug_dataset["test"]
drug_dataset_clean

### 保存数据集

Datasets 会缓存每个下载的数据集和对它执行的操作，但有时你会想要将数据集保存到磁盘。Datasets 提供了三个主要函数来以不同的格式保存你的数据集：

数据格式	对应的方法
Arrow	Dataset.save_to_disk()
CSV	    Dataset.to_csv()
JSON	Dataset.to_json()
例如，让我们以 Arrow 格式保存我们清洗过的数据集：
drug_dataset_clean.save_to_disk("drug-reviews")
这将创建一个具有以下结构的目录：
Copied
drug-reviews/
├── dataset_dict.json
├── test
│   ├── dataset.arrow
│   ├── dataset_info.json
│   └── state.json
├── train
│   ├── dataset.arrow
│   ├── dataset_info.json
│   ├── indices.arrow
│   └── state.json
└── validation
    ├── dataset.arrow
    ├── dataset_info.json
    ├── indices.arrow
    └── state.json
其中，我们可以看到，每个部分都有 dataset.arrow 表，以及保存元数据的 dataset_info.json 和 state.json 。你可以将 Arrow 格式视为一个优化过的列和行的精美表格，它针对构建处理和传输大型数据集的高性能应用程序进行了优化。

保存数据集后，我们可以使用 load_from_disk() 功能从磁盘读取数据：
from datasets import load_from_disk
drug_dataset_reloaded = load_from_disk("drug-reviews")

对于 CSV 和 JSON 格式，我们必须将每个部分存储为单独的文件。一种方法是遍历 DatasetDict 中的键和值
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
这将把每个部分保存为 JSON Lines格式 ，其中数据集中的每一行都存储为一行 JSON。

然后我们可以使用 第二节 中的技巧，按如下所示加载 JSON 文件
data_files = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl",
}
drug_dataset_reloaded = load_dataset("json", data_files=data_files)

## Datasets 应对大数据！
Datasets通过将数据集作为内存映射(memory-mapped)文件来处理，解放内存管理问题；并通过 流式处理(streaming) 来摆脱硬盘限制。

在本节中，我们将使用一个庞大的 825 GB 语料库——被称为 the Pile 的数据集，来探索🤗 Datasets 的这些功能。

### 什么是 the Pile？
The Pile 是由 EleutherAI 创建的一个用于训练大规模语言模型的英语文本语料库。它包含各种各样的数据集，涵盖科学文章，GitHub 代码库以及过滤后的 Web 文本。训练语料库以 14 GB 的文件块 提供，并且你也可以下载几个 单独的组件 。让我们先来看看 PubMed Abstracts 部分，它是 PubMed 上的 1500 万篇生物医学出版物的摘要的语料库。数据集采用 JSON Lines格式 并使用 zstandard 库进行压缩，所以我们

首先需要先安装 zstandard 库：
!pip install zstandard

加载远程数据集：
# 注意: the-eye.eu 已不可用，原 URL (https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst) 已失效。
# 以下使用 HuggingFace 镜像作为替代。
from datasets import load_dataset
data_files = "https://huggingface.co/datasets/casinca/PUBMED_title_abstracts_2019_baseline/resolve/main/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
pubmed_dataset = load_dataset("json", data_files=data_files, split="train")
pubmed_dataset

让我们看看数据集的第一个元素的内容：
pubmed_dataset[0]
{'meta': {'pmid': 11409574, 'language': 'eng'},
 'text': 'Epidemiology of hypoxaemia in children with acute lower respiratory infection.\nTo determine the prevalence of hypoxaemia in children aged under 5 years suffering acute lower respiratory infections (ALRI), the risk factors for hypoxaemia in children under 5 years of age with ALRI, and the association of hypoxaemia with an increased risk of dying in children of the same age ...'}
可以看到，这看起来像是医学文章的摘要。现在，让我们看看加载数据集所使用的 RAM！

### 内存映射
测量 Python 内存使用的简单方式是使用 psutil 库，可以通过如下方式安装：
!pip install psutil
它提供了一个 Process 类，让我们可以检查当前进程的内存使用情况，如下所示：
import psutil
Process.memory_info 是以字节为单位的,所以转换为兆字节
print(f"使用的RAM:{psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

这里的 rss 属性是指 常驻集（resident set size） 的大小，它是进程在 RAM 中占用的内存的部分。这个测量结果也包括了 Python 解释器和我们加载的库所使用的内存，所以实际上用于加载数据集的内存会更小一些。作为比较，让我们使用 dataset_size 属性看看数据集在磁盘上上的大小。由于结果像之前一样以字节为单位，我们需要手动将其转换为 GB：
print(f"数据集中文件的数量 : {pubmed_dataset.dataset_size}")
size_gb = pubmed_dataset.dataset_size / (1024**3)
print(f"数据集大小 (缓存文件) : {size_gb:.2f} GB")


如果你熟悉 Pandas，这个结果可能会让人感到很惊奇。因为根据 Wes Kinney 的著名的 经验法则 ，你通常需要 5 到 10 倍于你数据集大小的 RAM。那么 Datasets 是如何解决这个内存管理问题的呢？Datasets 将每一个数据集看作一个 内存映射文件 ，它提供了 RAM 和文件系统存储之间的映射，该映射允许 Datasets 库无需将其完全加载到内存中即可访问和操作数据集的元素。

内存映射文件也一个在多个进程之间共享，这使得像 Dataset.map() 之类的方法可以在无需移动或者复制数据集的情况下实现并行化。在底层，这些功能都是由 Apache Arrow 内存格式和 pyarrow 库实现的，这使得数据加载和处理速度快如闪电。为了更清晰地看到这个过程，让我们通过遍历 PubMed 摘要数据集中的所有元素，运行一个小速度测试：
import timeit
code_snippet = """batch_size = 1000
for idx in range(0, len(pubmed_dataset), batch_size):
    _ = pubmed_dataset[idx:idx + batch_size]


### 流式数据集
要使用数据集流，你只需要将 streaming=True 参数传递给 load_dataset() 函数。接下来，让我们以流模式加载 PubMed 摘要数据集：
pubmed_dataset_streamed = load_dataset(
    "json", data_files=data_files, split="train", streaming=True
)
不同于我们在这一章其它地方遇到的熟悉的 Dataset ， streaming=True 返回的对象是一个 IterableDataset 。顾名思义，要访问 IterableDataset ，我们需要迭代它。我们可以按照如下方式访问流式数据集的第一个元素：
next(iter(pubmed_dataset_streamed))

如果你需要在训练期间对流式数据集中的元素 tokenize，可以使用 IterableDataset.map() 进行在线处理，而不需要等待数据集全部加载完毕。该过程与我们在 第三章 中对数据集 tokenize 的过程完全相同，唯一的区别是输出是逐个返回的：
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = pubmed_dataset_streamed.map(lambda x: tokenizer(x["text"]))
next(iter(tokenized_dataset))

为了加速流式的 tokenize，你可以传递 batched=True ，就像我们在上一节看到的那样。它会批量处理示例；默认的批大小是 1000，可以通过 batch_size 参数指定批量大小。

使用 IterableDataset.shuffle() 打乱流式数据集，但与 Dataset.shuffle() 不同的是这只会打乱预定义 buffer_size 中的元素：
shuffled_dataset = pubmed_dataset_streamed.shuffle(buffer_size=10_000, seed=42)
next(iter(shuffled_dataset))

在这个例子中，我们从缓冲区的前 10,000 个示例中随机选择了一个示例。一旦访问了一个示例，它在缓冲区中的位置就会被语料库中的下一个示例填充 （即，上述案例中的第 10,001 个示例）。你还可以使用 IterableDataset.take() 和 IterableDataset.skip() 函数从流式数据集中选择元素，它的作用类似于 Dataset.select() 。例如，要选择 PubMed Abstracts 数据集的前 5 个示例，我们可以执行以下代码：
dataset_head = pubmed_dataset_streamed.take(5)
list(dataset_head)

同样，你可以使用 IterableDataset.skip() 函数从打乱的数据集中创建训练集和验证集，如下所示：
跳过前 1,000 个示例 ,将其余部分创建为训练集
train_dataset = shuffled_dataset.skip(1000)
将前 1,000 个示例用于验证集
validation_dataset = shuffled_dataset.take(1000)

让我们用一个常见的任务来进行我们对数据集流的最后探索：将多个数据集组合在一起创建一个新的语料库。Datasets 提供了一个 interleave_datasets() 函数，它将一个 IterableDataset 对象列表组合为单个的 IterableDataset ，其中新数据集的元素是交替抽取列表中的数据集获得的。当你试图组合大型数据集时，这个函数特别有用，让我们通过下面这个例子来试着组合 Pile 的 FreeLaw 数据集，这是一个包含美国法院法律意见的 51 GB 数据集：

law_dataset_streamed = load_dataset(
    "json",
    data_files="https://the-eye.eu/public/AI/pile_preliminary_components/FreeLaw_Opinions.jsonl.zst",
    split="train",
    streaming=True,
)
# 注意: the-eye.eu 已不可用，上述URL无法访问。暂无公开镜像，可尝试从 HuggingFace 搜索替代数据集。
next(iter(law_dataset_streamed))

这个数据集足够大，可以对大多数笔记本电脑的 RAM 有足够的压力，但是我们已经能够毫不费力地加载和访问它！现在我们使用 interleave_datasets() 函数将 FreeLaw 和 PubMed Abstracts 数据集的样本整合在一起：
from itertools import islice
from datasets import interleave_datasets

combined_dataset = interleave_datasets([pubmed_dataset_streamed, law_dataset_streamed])
list(islice(combined_dataset, 2))

这里我们使用了来自 Python 的 itertools 模块的 islice() 函数从合并的数据集中选择前两个示例，并且我们可以看到它们实际上就是两个源数据集中的前两个示例拼在一起形成的

最后，如果你想流式传输整个 825GB 的 Pile，你可以按照如下方式获取所有的预处理文件：
base_url = "https://the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
# 注意: the-eye.eu 已不可用，上述全部URL无法访问。替代方案：HuggingFace Hub 上搜索 "EleutherAI/pile" 或使用
# load_dataset("EleutherAI/pile", streaming=True) 直接从Hub流式加载（需接受条款）。
next(iter(pile_dataset["train"]))


## 创建自己的数据集
有时，不存在现有的合适的数据集适用于你构建 NLP 应用，因此你需要自己创建。在本节中，我们将向你展示如何创建一个由 GitHub issues 组成的的语料库，这些 issues 通常用于跟踪 GitHub 仓库中的错误或功能。该语料库可用于各种应用场景，包括：
探索解决 issue 或合并 pull 请求需要多长时间
训练一个 多标签分类器（multilabel classifier） 可以根据 issue 的描述为 issue 标上元数据标签（例如，“bug”、“enhancement（增强功能）”或“question”）
创建语义搜索引擎以查找与用户查询匹配的 issue
在这里，我们将关注如何创建语料库，在下一节中，我们将探索语义搜索。我们将使用一个流行的开源项目的 GitHub issue：🤗 Datasets！接下来让我们看看如何获取数据并探索这些 issue 中包含的信息。

### 获取数据
我们可以使用 GitHub REST API 遍历 Issues 节点(endpoint) 下载所有仓库的 issue。节点（endpoint）将返回一个 JSON 对象列表，每个对象包含大量字段，其中包括标题和描述以及有关 issue 状态的元数据等。

我们将使用 requests 库来下载，这是用 Python 中发出 HTTP 请求的标准方式。你可以通过运行以下的代码来安装 requests 库：
!pip install requests
安装库后，你通过调用 requests.get() 功能来获取 Issues 节点（endpoint）。例如，你可以运行以下命令来获取第一页上的第一个 Issues：
import requests
url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=1"
response = requests.get(url)
response.status_code
# 注意: 未认证请求限速 60次/小时，认证后 5000次/小时。使用 .env + python-dotenv 管理 token:
#   from dotenv import load_dotenv; import os; load_dotenv()
#   headers = {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"}
#   response = requests.get(url, headers=headers)

其中， 200 状态表示请求成功。不过，我们真正感兴趣的是消息体中的有效信息，由于我们知道我们的 issues 是 JSON 格式，让我们按如下方式查看消息体的信息：
response.json()

现在我们可以调用 fetch_issues() 它将按批次下载所有 issue，以避免超过 GitHub 每小时请求次数的限制；结果将存储在 repository_name-issues.jsonl 文件，其中每一行都是一个 JSON 对象，代表一个 issue。
fetch_issues()

下载 issue 后，我们可以使用我们在 第二节 新学会的方法在本地加载它们：
issues_dataset = load_dataset("json", data_files="datasets-issues.jsonl", split="train")
issues_dataset

### 清洗数据
pull_request 列可用于区分 issue 和 pull 请求。让我们随机挑选一些样本，看看有什么不同。Dataset.shuffle() 和 Dataset.select() 抽取一个随机样本，然后将 html_url 和 pull_request 列使用 zip 函数组合起来，以便我们可以比较各种 URL：
sample = issues_dataset.shuffle(seed=666).select(range(3))
打印出 URL 和 pull 请求
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")

这里我们可以看到，每个 pull 请求都与各种 url 相关联，而普通 issue 只有一个 None 条目。我们可以使用这一点不同来创建一个新的 is_pull_request 列，通过检查 pull_request 字段是否为 None 来区分它们：
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

### 扩充数据集
issue 或 pull 请求相关的评论提供了丰富的信息，特别是如果我们有兴趣构建搜索引擎来回答用户对这个仓库的疑问时，这些信息将非常有用。
GitHub REST API 提供了一个 Comments(评论)节点 返回与 issue 编号相关的所有评论。让我们测试一下该节点返回的内容：
issue_number = 2792
url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
response = requests.get(url, headers=headers)
response.json()

评论存储在 body 字段中，因此让我们编写一个简单的函数，通过挑选出 response.json() 中每个元素的 body 内容，返回与某个 issue 相关的所有评论：

def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

get_comments(2792)

使用 Dataset.map() 方法，为我们数据集中每个 issue 的添加一个 comments 列：
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)

### 将数据集上传到 Hugging Face Hub

使用 push_to_hub() 方法来推送数据集。为此，我们需要一个身份验证令牌，它可以通过首先使用 notebook_login() 函数登录到 Hugging Face Hub 来获得：
from huggingface_hub import notebook_login
notebook_login()

这将创建一个小部件，你可以在其中输入你的用户名和密码，API 令牌将保存在 ~/.huggingface/token 中。如果你在终端中运行代码，则可以改为使用命令行登录：
huggingface-cli login

完成此操作后，我们可以通过运行下面的代码上传我们的数据集：
issues_with_comments_dataset.push_to_hub("github-issues")

之后，任何人都可以通过便捷地使用附带仓库 ID 作为 path 参数的 load_dataset() 函数 来下载数据集：
remote_dataset = load_dataset("lewtun/github-issues", split="train")
remote_dataset

### 创建数据集卡片
有据可查的数据集更有可能对其他人（包括你未来的自己！）有用，因为它们提供了数据集相关的信息，使用户能够决定数据集是否与他们的任务相关，并评估任何潜在的偏见或与使用相关的风险。

在 Hugging Face Hub 上，此信息存储在每个数据集仓库的自述文件（README.md）。在创建此文件之前，你应该采取两个主要步骤：

使用 datasets-tagging应用程序 创建 YAML 格式的元数据标签。这些标签用于 Hugging Face Hub 上的各种搜索功能，并确保你的数据集可以很容易地被社区成员找到。由于我们已经在这里创建了一个自定义数据集，所以你需要克隆数据集标签仓库( datasets-tagging )并在本地运行应用程序。它的界面是这样的：

2．阅读 🤗 Datasets 指南 中关于创建完善的数据集卡片的指南，并将其作为模板使用。
你可以直接在 Hub 上创建 README.md 文件，你可以在 lewtun/github-issues 数据集仓库中找到一个模板数据集卡片。下面显示了填写好的数据集卡片的截图。



## 使用 FAISS 进行语义搜索

### 使用文本嵌入进行语义搜索
基于 Transformer 的语言模型会将文本中的每个 token 转换为嵌入向量。事实证明，我们可以 “池化（pool）” 嵌入向量以创建整个句子、段落或（在某些情况下）文档的向量表示。然后，通过计算每个嵌入之间的点积相似度（或其他一些相似度度量）并返回相似度最大的文档，这些嵌入可用于在语料库中找到相似的文档。

在本节中，我们将使用文本嵌入向量来开发语义搜索引擎。与基于将查询中的关键字的传统方法相比，这些搜索引擎具有多种优势。


### 加载和准备数据集
首先，我们需要下载我们的 GitHub issues 数据集，所以让我们像往常那样使用 load_dataset() 函数：
from datasets import load_dataset
issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset

在此，我们在 load_dataset() 中指定了默认的 train（训练集） 部分，因此它返回一个 Dataset 而不是 DatasetDict 。首要任务是排除掉拉取请求（pull），因为这些请求往往很少用于回答提出的 issue，会为我们的搜索引擎引入干扰。正如我们现在熟悉的那样，我们可以使用 Dataset.filter() 函数来排除数据集中的这些行。与此同时，让我们也筛选掉没有评论的行，因为这些行没有为用户提问提供回答：
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
issues_dataset

我们可以看到，我们的数据集中有很多列，其中大部分在构建我们的搜索引擎都不会使用。从搜索的角度来看，信息量最大的列是 title ， body ，和 comments ，而 html_url 为我们提供了一个回到原 issue 的链接。让我们使用 Dataset.remove_columns() 删除其余的列：

columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset

为了创建我们的文本嵌入数据集，我们将用 issue 的标题和正文来扩充每条评论，因为这些字段通常包含有用的上下文信息。因为我们的 comments 列当前是每个 issue 的评论列表，我们需要“重新组合”列，使得每一行都是由一个 (html_url, title, body, comment) 元组组成。在 Pandas 中，我们可以使用 DataFrame.explode() 函数 完成这个操作 它为类似列表的列中的每个元素创建一个新行，同时复制所有其他列值。让我们首先切换到 Pandas 的 DataFrame 格式：
issues_dataset.set_format("pandas")
df = issues_dataset[:]

如果我们检查这个 DataFrame 的第一行，我们可以看到这个 issue 有四个相关评论：

我们希望使用 explode() 将这些评论中的每一条都展开成为一行。让我们看看是否可以做到：
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)

非常好，我们可以看到其他三列已经被复制了，并且 comments 列里存放着单独的评论！现在我们已经完成了 Pandas 要完成的部分功能，我们可以通过加载内存中的 DataFrame 快速切换回 Dataset ：

from datasets import Dataset
comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset


看看你是否可以使用 Dataset.map() 展开 issues_dataset 的 comments 列，这有点棘手；你可能会发现 🤗 Datasets 文档的 “批处理映射(Batch mapping)” 对这个任务很有用。

既然我们每行有一个评论，让我们创建一个新的 comments_length 列来存放每条评论的字数：
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)

我们可以使用这个新列来过滤掉简短的评论，其中通常包括“cc @lewtun”或“谢谢！”之类与我们的搜索引擎无关的内容。筛选的精确数字没有硬性规定，但大约大于 15 个单词似乎是一个不错的选择：
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset

稍微清理了我们的数据集后，让我们使用 issue 标题、描述和评论构建一个新的 text 列。像往常一样，我们可以编写一个简单的函数，并将其传递给 Dataset.map() 来完成这些操作
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }

comments_dataset = comments_dataset.map(concatenate_text)

### 创建文本嵌入
可以通过使用 AutoModel 类来完成文本嵌入。首先需要做的就是选择一个合适的 checkpoint。幸运的是，有一个名为 sentence-transformers 的库专门用于创建文本嵌入。如库中的 文档 所述的，我们这次要实现的是非对称语义搜索（asymmetric semantic search），因为我们有一个简短的查询，我们希望在比如 issue 评论等更长的文档中找到其匹配的文本。通过查看 模型概述表 我们可以发现 multi-qa-mpnet-base-dot-v1 checkpoint 在语义搜索方面具有最佳性能，因此我们将使用它。我们还将使用这个 checkpoint 加载了对应的 tokenizer ：

from transformers import AutoTokenizer, AutoModel
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

将模型和输入的文本放到 GPU 上会加速嵌入过程，所以让我们现在就这么做：

import torch
device = torch.device("cuda")
model.to(device)

根据我们之前的想法，我们希望将我们的 GitHub issue 中的每一条记录转化为一个单一的向量，所以我们需要以某种方式“池化（pool）”或平均每个词的嵌入向量。一种流行的方法是在我们模型的输出上进行 CLS 池化 ，我们只需要收集 [CLS] token 的的最后一个隐藏状态。以下函数实现了这个功能：

def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

接下来，我们将创建一个辅助函数，它将对一组文档进行 tokenize，然后将张量放在 GPU 上，接着将它们喂给模型，最后对输出进行 CLS 池化：

def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

我们可以将第一条数据喂给它并检查输出的形状来测试这个函数是否正常工作：
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape

太好了，我们已经将语料库中的第一个条目转换为了一个 768 维向量！我们可以用 Dataset.map() 将我们的 get_embeddings() 函数应用到我们语料库中的每一行，然后创建一个新的 embeddings 列：

# 批处理版本(推荐): 比逐条处理快得多，batch_size=32 是GPU上的常用值
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": list(get_embeddings(x["text"]))},
    batched=True,
    batch_size=32,
)
# 同时应在 get_embeddings 内加 torch.no_grad() 以节省显存:
#   def get_embeddings(text_list):
#       encoded_input = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
#       encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
#       with torch.no_grad():
#           model_output = model(**encoded_input)
#       return cls_pooling(model_output)

### 使用 FAISS 进行高效的相似性搜索
现在我们有了一个文本嵌入数据集，我们需要一些方法来搜索它们。为此，我们将使用🤗 Datasets 中一种特殊的数据结构，称为 FAISS 指数。 FAISS （Facebook AI Similarity Search 的缩写）是一个库，提供了用于快速搜索和聚类嵌入向量的高效算法。

FAISS 背后的基本思想是创建一个特殊的数据结构，称为 index（索引） 它可以找到哪些嵌入与输入嵌入相似。在 🤗 Datasets 中创建一个 FAISS index（索引）很简单——我们使用 Dataset.add_faiss_index() 函数并指定我们要索引的数据集的哪一列：

embeddings_dataset.add_faiss_index(column="embeddings")

现在，我们可以使用 Dataset.get_nearest_examples() 函数进行最近邻居查找。让我们通过首先嵌入一个 issue 来测试这一点，如下所示：

question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape

就像对文档进行嵌入一样，我们现在有一个表示查询的 768 维向量，我们可以将其与整个语料库进行比较以找到最相似的嵌入：

scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
Dataset.get_nearest_examples() 函数返回一个元组，包括评分（评价查询和文档之间的相似程度）和对应的样本（这里是 5 个最佳匹配）。让我们把这些收集到一个 pandas.DataFrame ，这样我们就可以轻松地对它们进行排序：

import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

现在我们可以遍历前几行来查看我们的查询与评论的匹配程度如何：

for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()


## 今日知识总结

### 学习曲线：判断模型健康状况

学习曲线是训练过程中损失和准确率随时间变化的可视化。损失曲线应平稳下降，准确率曲线应阶梯式上升。过拟合的特征是训练损失继续下降而验证损失反弹，欠拟合则是两条线都居高不下。不稳定曲线通常意味着学习率过大或批大小过小。学习曲线的作用是回答"模型还要不要继续训练"——而不是准确率本身。

### Datasets 库：三种加载方式

Datasets 加载数据有三种路径：本地文件、远程 URL、Hugging Face Hub。核心是 `load_dataset()` + `data_files` 参数。本地 JSON/CSV 直接用文件路径；远程用完整 URL，库自动下载解压；Hub 上的直接给 repo ID。`data_files` 可以是单文件、列表，或 `{"train": "...", "test": "..."}` 字典——字典形式返回带多个 split 的 `DatasetDict`。

### 数据处理三板斧：filter、map、列操作

`Dataset.filter()` 删除不需要的行（如 condition 为 None 的记录），`Dataset.map()` 批量改数据（tokenization、lowercase、去 HTML），`rename_column` / `remove_columns` 改列结构。`map()` 的 `batched=True` 是关键加速开关——快速 tokenizer 配合批处理可快 30 倍。`num_proc` 对 Rust tokenizer 帮助有限，但在 CPU 密集型操作中有用。Windows 下 `num_proc` 用 `spawn` 方式创建子进程，不继承全局变量，不适用于网络请求。

### 内存映射与流式处理：应对大数据

Datasets 用 Apache Arrow 内存映射，数据集不需要全部加载到 RAM，实际内存占用远小于文件大小。流式处理 (`streaming=True`) 返回 `IterableDataset`，用 `take()` / `skip()` 选取样本，`interleave_datasets()` 合并多个流。适合 825GB 的 The Pile 这种规模。

### GitHub API 认证与限流

未认证请求每小时只有 60 次，认证后 5000 次。用 `python-dotenv` + `.env` 文件管理 token，代码中 `load_dotenv()` 加载，headers 传 `Authorization: Bearer <token>`。注意 notebook 工作目录在 `notebooks/`，`.env` 需在同目录或指定路径。JSONL 加载时箭头 schema 推断可能与嵌套字段冲突，用 `pd.read_json()` + `Dataset.from_pandas()` 过渡。

### 语义搜索：嵌入 + FAISS

语义搜索用 Transformer 生成文本嵌入向量，比较查询和文档的向量相似度。`[CLS]` pooling 是对整个句子取 `last_hidden_state[:, 0]` 作为固定长度的句向量。FAISS 是做向量相似度检索的库，`add_faiss_index()` 建索引，`get_nearest_examples()` 找最相似文档。嵌入生成用 `batched=True` + `batch_size=32` 可以在 GPU 上高效并行，单条处理会慢很多。Windows 只有 `faiss-cpu` 没有 `faiss-gpu`。

### 今日解决的问题

- `the-eye.eu` 挂了 → PubMed 数据集用 HuggingFace mirror (`casinca/PUBMED_title_abstracts_2019_baseline`)
- UCI 药物数据集地址失效 → 用 HuggingFace 或 Kaggle 替代源
- GitHub API `403 rate limit` → token 认证 + `.env` 管理
- Windows `wget` 只是 `Invoke-WebRequest` 别名 → 需加 `-OutFile` 才存盘
- `NameError: name 'headers' is not defined` → `fetch_issues` 函数内定义 headers，用 `dotenv` 加载 token
- `TypeError` JSONL schema 不匹配 → pandas 先读再转 Dataset
- Windows `num_proc` spawn 子进程丢失全局变量 → HTTP 请求不设多进程
- `paiss-gpu` Windows 不支持 → 装 `faiss-cpu`
- `get_embeddings` map 单条卡住 → 改为 `batched=True, batch_size=32`，加 `torch.no_grad()`

### 下一步

Day 08：进入 Week 2，开始 PyTorch 基础训练——Tensor 操作、Dataset/DataLoader、nn.Module、autograd、手写训练循环。

