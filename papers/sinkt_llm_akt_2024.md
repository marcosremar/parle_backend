## **SINKT: A Structure-Aware Inductive Knowledge Tracing Model** **with Large Language Model**



Kounianhua Du

Shanghai Jiao Tong University
Shanghai, China
kounianhuadu@sjtu.edu.cn


Weinan Zhang
Shanghai Jiao Tong University
Shanghai, China
wnzhang@sjtu.edu.cn


Yong Yu [∗]

Shanghai Jiao Tong University
Shanghai, China
yyu@apex.sjtu.edu.cn



Lingyue Fu
Shanghai Jiao Tong University
Shanghai, China
fulingyue@sjtu.edu.cn


Jianghao Lin
Shanghai Jiao Tong University
Shanghai, China
chiangel@sjtu.edu.cn


Ruiming Tang
Huawei Noah’s Ark Lab

Shenzhen, China
tangruiming@huawei.com


**ABSTRACT**



Hao Guan

Shanghai Jiao Tong University
Shanghai, China
Guanhao2022@sjtu.edu.cn


Wei Xia

Huawei Noah’s Ark Lab

Shenzhen, China
xiawei24@huawei.com


Yasheng Wang
Huawei Noah’s Ark Lab

Shenzhen, China
wangyasheng@huawei.com


**CCS CONCEPTS**



Knowledge Tracing (KT) aims to determine whether students will
respond correctly to the next question, which is a crucial task in
intelligent tutoring systems (ITS). In educational KT scenarios,
transductive ID-based methods often face severe data sparsity and
cold start problems, where interactions between individual students
and questions are sparse, and new questions and concepts consistently arrive in the database. In addition, existing KT models only
implicitly consider the correlation between concepts and questions,
lacking direct modeling of the more complex relationships in the
heterogeneous graph of concepts and questions. In this paper, we
propose a Structure-aware INductive Knowledge Tracing model
with large language model (dubbed **SINKT** ), which, for the first
time, introduces large language models (LLMs) and realizes inductive knowledge tracing. Firstly, SINKT utilizes LLMs to introduce
structural relationships between concepts and constructs a heterogeneous graph for concepts and questions. Secondly, by encoding
concepts and questions with LLMs, SINKT incorporates semantic
information to aid prediction. Finally, SINKT predicts the student’s
response to the target question by interacting with the student’s
knowledge state and the question representation. Experiments on
four real-world datasets demonstrate that SINKT achieves state
of-the-art performance among 12 existing transductive KT models.
Additionally, we explore the performance of SINKT on the inductive
KT task and provide insights into various modules.


∗The corresponding author.


Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
_CIKM ’24, October 21–25, 2024, Boise, ID, USA._
© 2024 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-0436-9/24/10
[https://doi.org/10.1145/3627673.3679760](https://doi.org/10.1145/3627673.3679760)




- **Social and professional topics** → **Student assessment** ; • **Ap-**
**plied computing** → **E-learning** .


**KEYWORDS**


Knowledge Tracing, Inductive Learning, Online Education


**ACM Reference Format:**

Lingyue Fu, Hao Guan, Kounianhua Du, Jianghao Lin, Wei Xia, Weinan
Zhang, Ruiming Tang, Yasheng Wang, and Yong Yu. 2024. SINKT: A StructureAware Inductive Knowledge Tracing Model with Large Language Model.
In _Proceedings of the 33rd ACM International Conference on Information and_
_Knowledge Management (CIKM ’24), October 21–25, 2024, Boise, ID, USA._ ACM,
[New York, NY, USA, 11 pages. https://doi.org/10.1145/3627673.3679760](https://doi.org/10.1145/3627673.3679760)


**1** **INTRODUCTION**


Intelligent tutoring systems (ITS), such as Massive Open Online
Courses (MOOCs) [1] and Khan Academy [2], are garnering increasing
attention from learners due to the extensive learning resources and
real-time feedback mechanisms. Knowledge Tracing (KT) stands
out as a vital area of research within ITS, which aims to assess the
current knowledge states of students by analyzing their learning
histories and to predict their responses to upcoming questions.
KT not only helps to identify deficiencies in student’s learning
content but also provides a foundational basis for the subsequent
instructional strategies in ITS.
Existing KT models originate from traditional methods represented by BKT [7] and have evolved significantly with the advent
of models based on deep neural networks. Deep learning-based KT
models learn ID embeddings for questions and concepts through
learning history of students. DKT [34] and DHKT [42] utilize Recurrent Neural Networks (RNNs) [35] to capture the sequential
information of learning history. EERNNA [37], SAKT [32], and
AKT [12] employ attention mechanisms to gauge the importance
of a student’s learning history in addressing the current question.


1https://www.mooc.org/
2https://www.khanacademy.org/


CIKM ’24, October 21–25, 2024, Boise, ID, USA. Fu et al.





**Student A**





**Figure 1: A demonstration of duplicated relations between**
**concepts and questions in ITS. Concept** _𝑐_ 4 **and question** _𝑞_ 5
**are newly added to ITS, which do not appear in any learning**
**history.**


SKVMN [3] and DKVMN [50] apply memory networks [43] to
exploit the relationships between concepts and questions. Some
models [2, 28, 30, 45, 48] introduce additional information from the
learning process to assist response prediction. IEKT [28] selects
learning histories of similar students to enhance the understanding
of the current student’s knowledge state. GKT [30] and GIKT [48]
incorporate the transition graph originated from learning history
and use Graph Neural Networks (GNNs) [17] to extract representations from graph data. CL4KT [2] designs positive and negative
sample pairs and employs contrastive learning [13] to generate
more accurate representations of questions. LBKT [45] introduces
various learning behaviors, including speed, attempts, and hints, to
identify the learning patterns of students. Despite the success of
existing KT models, some important limitations still exist.
Firstly, current ID-based transductive KT models cannot give
accurate predictions to newly arrived questions and questions with
sparse interactions. _Transductive KT models_ learn ID embeddings of
questions and concepts from learning history and predict responses
to those questions that have appeared in the dataset. Existing KT
models are all ID-based and are therefore considered transductive

KT models. However, as illustrated in Figure 1, when a new concept
_𝑐_ 4 and a new question _𝑞_ 5 are added to the database of ITS, transductive KT models lack interaction data to learn their representations,
thus unable to predict the response of the student A. Models that
could predict those questions that haven’t appear in the dataset are
called _inductive KT models_ . Meanwhile, interactions with individual
students are limited, and thus question-level information is very
sparse, which is difficult for ID-based transductive KT models to
train. For example, the commonly used MovieLens dataset [3] in
recommendation systems contains 25,000,095 ratings and 1,093,360
tag applications across 62,423 movies. However, in educational
datasets, such as ASSIST09 [4], there are only 2,661 students with
165,455 interactions, while the total number of questions is 14,083.
Secondly, current methods fail to capture the semantic dependencies and the sufficient topological relations among questions


3https://grouplens.org/datasets/movielens/
4https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data



and concepts. For example, as shown in Figure 1, _𝑞_ 1 and _𝑞_ 2 are
computational questions, while _𝑞_ 3 and _𝑞_ 4 are application questions. Compared to computational questions, application questions
require students to possess additional analytical skills and have
higher cognitive demands. Additionally, structural information exists among concepts. For instance, addition _𝑐_ 1 should serve as the
foundation for subtraction _𝑐_ 2 and multiplication _𝑐_ 3 during the learning process. If a student struggles with the concept “addition”, he
will likely have poor demand for concepts “subtraction” and “multiplication”. Directly learning from learning sequences using ID
embeddings makes it difficult to capture such type of information.
In ITS, structural and semantic information are indispensable, as
they can provide additional clues for knowledge tracing.
Based on these considerations, in this paper, we propose a novel
Structure-Aware INductive Knowledge Tracing Model with Large
Language Model, named **SINKT** . To the best of our knowledge,
SINKT is the first work that can achieve predictions for newly
ingested questions, _i.e.,_ realize inductive knowledge tracing. Our
framework also integrates open-world semantic and structural information by LLMs for the first time, enabling it to dynamically
update and expand its knowledge base with minimal manual intervention. Specifically, SINKT first employs LLMs to generate a
concept-question heterogeneous graph that contains useful structural information in ITS. Then, we use a Pretrained Language Model
(PLM) to encode semantic information of questions and concepts
instead of training ID embeddings. Subsequently, we encode the
concept-question graph by applying a carefully designed structural
information encoder. Finally, we design a student state encoder
to capture sequential information and an interaction predictor to
predict the student’s response with his knowledge state and the
target question description.
To sum up, our contributions are summarized as follows:


- We propose a novel Structure-Aware Inductive Knowledge Tracing model with LLM ( _i.e.,_ SINKT), and a full-automated pipeline
for ITS to inductively trace mastery of new concepts and predict
response to new questions, which is the first model to realize
inductive knowledge tracing.

 - We utilize LLMs for the first time to carefully inject open-world
knowledge, including semantic and structural information of
questions and concepts, into the KT task.

- Extensive experiments on four real-world datasets demonstrate
that SINKT achieves state-of-the-art performance against 12
baselines, as well as superior capabilities to predict responses
for unseen questions.


**2** **RELATED WORK**

**2.1** **Knowledge Tracing**


To investigate the learning patterns of students in ITS, numerous
KT models have been proposed to encode knowledge and model interaction histories. Traditional KT models such as Bayesian Knowledge Tracing [8] focus on tracing students’ knowledge states by
estimating general parameters. With the advent of deep learning,
DKT [34] is the first to utilize RNN [35] and LSTM [15] to model
the sequential dynamics of student interactions. DKT+ [49] and
DHKT [42] extend the implementation of DKT, taking into account
the relationships between questions and concepts. Inspired by the


SINKT: A Structure-Aware Inductive Knowledge Tracing Model
with Large Language Model CIKM ’24, October 21–25, 2024, Boise, ID, USA.



Transformer architecture [41], attention mechanisms [5] have been
incorporated into deep learning KT models for capturing the relationships among questions and their relevance to a student’s
knowledge states [12, 32, 37].
Recently, attention has been paid to more nuanced behavioral
data and advanced neural architectures. For instance, LBKT [45]
analyzes a range of learning behaviors, containing the speed of
responses, the number of attempts, and the use of hints. SAKT [32]
and AKT [12] introduce attention mechanisms [5] to focus on the
most relevant aspects of a student’s past interactions. Additionally,
innovative approaches such as CMKT [52]assess students’ dynamic
mastery of concepts. MF-DAKT [51], which incorporates multiple
student-related factors with a dual attention mechanism, showcases the field’s advancement towards more granular and complex
models. Alongside these developments, graph-based KT models
like GKT [30] and GIKT [48] employ GNN [17] to leverage the
relational data among concepts.


**2.2** **LLM-Enhanced User Modeling**


There have been some works using LLMs to enhance the performance of user modeling [22, 38]. Some work [23–25, 47] apply
LLMs as profilers to involve the creation of prompts based on users’
history, and input these prompts into LLMs to generate various
aspects of user profiles. KAR [44] leverages LLMs to generate user
and item profiles, encompassing user preference and real-world
knowledge of items into click-through rate (CTR) prediction task.
GIRL [53] uses LLMs to generate suitable job descriptions base on
the user’s curriculum vitae to help recommendation model understand jobhunter’s preferences. In addition to profiling, LLMs serve
as feature encoders, translating raw user data into rich, semantic embeddings that improve user modeling systems [20, 33, 44].
GPT4SM [33] employs GPT to encode queries and candidate text for
relevance prediction in recommendation systems, and LKPNR [14]
uses open-source LLMs for better semantic representation of news
articles. Moreover, LLMs function as knowledge augmenters by
integrating external knowledge into user models [4, 10, 19, 21, 44],
thus enhancing the precision of user-related predictions. Mysore
et al. [29] augment narrative-driven recommendations by generating author narrative queries with LLMs, and KAR [44] uses LLMprompted outputs to augment recommender systems with factual
item knowledge.
Existing LLM-enhanced user modeling methods are mostly applied in the domain of recommender systems (RS), which differ
significantly from the KT task. In RS, users’ interests typically remain stable over short periods, whereas in ITS, the knowledge states
of students evolve dynamically as they engage with educational
contents such as exercises. Additionally, there is strong structural
information, such as interdependency and correlation among the
concepts in ITS. Therefore, there is a need for specially designed
models that can effectively track and adapt to the unique dynamics
of students’ learning processes in ITS.


**3** **PRELIMINARIES**

**3.1** **Problem Definition**


Consider a set of students S, a set of questions Q, and a set of
concepts C within an ITS. The learning history of a student _𝑠_ ∈S



is recorded as _𝑅𝑠_ = {( _𝑞_ 1 _,𝑟_ 1) _,_ ( _𝑞_ 2 _,𝑟_ 2) _, . . .,_ ( _𝑞𝑇_ _,𝑟𝑇_ )}, where _𝑞𝑡_ ∈Q
represents the question answered by the student at time step _𝑡_, and
_𝑟𝑡_ ∈{0 _,_ 1} indicates whether the student _𝑠_ correctly answers the
question _𝑞𝑡_ (1 for correct and 0 for incorrect). Given the learning
history _𝑅𝑠_ of a student and a new question _𝑞𝑇_ +1, the goal of the KT
task is to predict the probability that the student correctly answers
the new question, denoted as _𝑝_ ( _𝑟𝑇_ +1 = 1| _𝑅𝑠,𝑞𝑇_ +1).
Note that in a real ITS, teachers continually expand the question and concept set, meaning that both the question set Q and
the concept set C are in a state of ongoing growth. Previous KT
models are trained and tested under the assumption Q _𝑡𝑟𝑎𝑖𝑛_ = Q _𝑡𝑒𝑠𝑡_,
which are only suitable for the transductive KT task. However, in
the inductive KT task, Q _𝑡𝑟𝑎𝑖𝑛_ ≠ Q _𝑡𝑒𝑠𝑡_ _._ In this paper, our SINKT
is capable of addressing the inductive task effectively while also
delivering excellent performance in the transductive KT task.
**3.2** **Concept-Question Graph Generation**


To inject open-world knowledge into SINKT, we construct a heterogeneous concept-question graph G = (V _,_ E _,_ OV _,_ R E ) to capture
effective representations of concepts and questions, where OV
represent the set of vertex type, and R E denotes the edge type (relation). The vertex set V includes the question set Q and the concept
set C, and The vertex type set OV includes _concept_ and _question_ correspondingly. Edge set E describes different relationships between
concepts and questions. The edge type set R E includes _concept-_
_question_, _question-concept_ and _concept-concept_ . The graph G not
only encapsulates the relationships between questions and concepts contained in the dataset but also incorporates open-world
knowledge provided by LLMs.
_concept-question_ edges and _question-concept_ edges are derived
from the dataset. Each question _𝑞𝑖_ corresponds to one or more
concepts C _𝑞𝑖_ = { _𝑐_ 1 _,𝑐_ 2 _, . . .,𝑐𝑛𝑖_ }, and each concept _𝑐𝑖_ corresponds to
many questions Q _𝑐𝑖_ = { _𝑞_ 1 _,𝑞_ 2 _, . . .,𝑞𝑚𝑖_ }, where _𝑛𝑖_ and _𝑚𝑖_ denotes
the number of concepts related to the question _𝑞𝑖_ and the number
of questions related to the concept _𝑐𝑖_, respectively. In the graph G,
the relation is described as _𝑛𝑖_ _question-concept_ edges ⟨ _𝑞𝑖,𝑐_ _𝑗_ ⟩( _𝑗_ =
1 _,_ 2 _, . . .,𝑛𝑖_ ) and _𝑚𝑖_ _concept-question_ edges ⟨ _𝑐𝑖,𝑞_ _𝑗_ ⟩( _𝑗_ = 1 _,_ 2 _, . . .,𝑚𝑖_ ).
_concept-concept_ edges within the graph G are generated by LLM.
As demonstrated in Figure 2, for each concept _𝑐𝑖_, we employ a
specific prompt and utilize GPT-4 [31] to identify a list of related
concepts. [11, 18] suggest that LLMs struggle with tasks directly
involving graph structures, often requiring specialized training or
input transformation to handle graph structures. Therefore, we opt
to use a “select from the list” approach rather than asking GPT-4 to
directly generate the graph. After GPT-4 generates the response,
we use the regular expression extraction to get _𝑝𝑖_ concepts related
to _𝑐𝑖_, and add the directed edges ⟨ _𝑐_ _𝑗_ _,𝑐𝑖_ ⟩( _𝑗_ = 1 _,_ 2 _, . . ., 𝑝𝑖_ ) into the
graph G.


**4** **METHODOLOGY**


In this section, we will introduce our method in detail, and the overall framework is shown in Figure 3. After generating the conceptquestion graph by GPT-4, we first use the Textual Information
Encoder (TIEnc) to extract semantic information from the concepts
and questions. Then, we design a Structural Information Encoder
(SIEnc) to learn question and concept representations from the heterogeneous graph. Finally, we design a student state encoder based


CIKM ’24, October 21–25, 2024, Boise, ID, USA. Fu et al.











**Figure 2: Prompt demonstration of concept relation graph generation.**



on concept-level mastery to capture the student’s knowledge state,
and predicts the final response by an interaction predictor.


**4.1** **Textual Information Encoder**


The introduction of LLMs bridges the gap between semantic information and plain text. SINKT utilizes a Pretrained Language Model
(PLM) as semantic encoder to get representations of the plain text
of concepts and questions. For each concept _𝑐𝑖_, we input its plain
text TEXT( _𝑐𝑖_ ) into the PLM and acquire its representation vector

x _[𝑐]_ _𝑖_ [=][ PLM] _[𝑐]_ [(][TEXT][(] _[𝑐][𝑖]_ [)) ∈] [R] _[𝑑][𝑡]_ _[,]_ (1)


where _𝑑𝑡_ is the encoding dimension of the PLM. Similarly, for each
question _𝑞𝑖_, we could obtain its representation vector

x _[𝑞]_ _𝑖_ [=][ PLM] _[𝑞]_ [(][TEXT][(] _[𝑞][𝑖]_ [)) ∈] [R] _[𝑑][𝑡]_ _[.]_ (2)

The representation vectors x _[𝑐]_ _𝑖_ [and][ x] _[𝑞]_ _𝑖_ [contain the semantic infor-]
mation of concepts and questions. In our framework, we fix the
parameters of PLM encoder and use representation vectors as an
input of following modules.


**4.2** **Structural Information Encoder**


When a student answers a question, their mastery of concepts
associated with that question influence their response. The mastery of both the target concept and its related concepts, as well
as the related questions, can provide valuable information for the
assessment of KT models. Therefore, the three types of edges in
the heterogeneous graph G, _i.e.,_ concept-question, concept-concept,
and question-concept, are considered by SINKT. To encode such
structural information, we carefully design a multi-layer heterogeneous graph encoder.
In the graph G, denote neighbor questions of the concept _𝑐𝑖_
be N _𝑐_ _[𝑐]_ _𝑖_ and N _𝑐_ _[𝑞]_ _𝑖_ [, respectively. The set of neighbor concepts of the]
question _𝑞𝑖_ is denoted as N _𝑞_ _[𝑐]_ _𝑖_ . We apply three different Graph Attention Networks (GAT) to aggregate neighborhood information
for vertexes. Concept-Question GAT layer fuse neighbor concept
representations of the target concept _𝑐𝑖_ . Specifically, the ConceptConcept GAT can be expressed as:

exp �LeakyReLU �acc _[𝑇]_ (x _[𝑞]_ _𝑖_ [⊕] [x] _[𝑐]_ _𝑗_ [)] ��
_𝛼𝑖,𝑗_ _[𝑐𝑞]_ [=] _,_ (3)

~~�~~ _𝑐𝑘_ ∈N _𝑐𝑖_ _[𝑞]_ [exp] ~~�~~ LeakyReLU ~~�~~ acc _[𝑇]_ (x _[𝑞]_ _𝑖_ [⊕] [x] _[𝑐]_ _𝑘_ [)] ~~��~~



where Wcq is the aggregate weight a _𝑐𝑞_ is the attention weight, ⊕
denotes concatenate operation. Similarly, Concept-Concept GAT
integrate neighbor concept information to the target concept:

exp �LeakyReLU �acc _[𝑇]_ (x _[𝑐]_ _𝑖_ [⊕] [x] _[𝑐]_ _𝑗_ [)] ��
_𝛼𝑖,𝑗_ _[𝑐𝑐]_ [=] _,_ (5)

~~�~~ _𝑐𝑘_ ∈N _𝑐𝑖_ _[𝑐]_ [exp] ~~�~~ LeakyReLU ~~�~~ acc _[𝑇]_ (x _[𝑐]_ _𝑖_ [⊕] [x] _[𝑐]_ _𝑘_ [)] ~~��~~



e _[𝑐𝑐]_ _𝑖_ = ∑︁

_𝑐_ _𝑗_ ∈N _𝑐𝑖_ _[𝑐]_



_𝛼𝑖,𝑗_ _[𝑐𝑐]_ [∗] �W _𝑐𝑐_ e _[𝑐]_ _𝑗_ � _,_ (6)



and Question-Concept GAT integrate neighbor concept information
to the target concept:

exp �LeakyReLU �aqc _[𝑇]_ (x _[𝑐]_ _𝑖_ [⊕] [x] _[𝑞]_ _𝑗_ [)] ��
_𝛼𝑖,𝑗_ _[𝑞𝑐]_ [=] _,_ (7)

~~�~~ _𝑞𝑘_ ∈N _𝑞𝑖_ _[𝑐]_ [exp] ~~�~~ LeakyReLU ~~�~~ aqc _[𝑇]_ (x _[𝑐]_ _𝑖_ [⊕] [x] _[𝑞]_ _𝑘_ [)] ~~��~~



e _[𝑞𝑐]_ =
_𝑖_ ∑︁

_𝑞_ _𝑗_ ∈N _𝑞𝑖_ _[𝑐]_



_𝛼𝑖,𝑗_ _[𝑞𝑐]_ [∗] �W _𝑞𝑐_ x _[𝑞]_ _𝑗_ � _._ (8)



To emphasize the semantic information inherent in concepts and
questions themselves, SINKT introduces jumping knowledge [46]
to propagate the vertex original representations. The node representation of _𝑙_ -th encoder layer of can be represented as:

x _[𝑐]_ _𝑖_ [(] _[𝑙]_ [)] = ReLU �Wcx _[𝑐]_ [(] _[𝑙]_ [−][1][)] + e _[𝑐𝑐]_ _𝑖_ + e _[𝑞𝑐]_ _𝑖_ � _,_ (9)


x _[𝑞]_ _𝑖_ [(] _[𝑙]_ [)] = ReLU �Wqx _[𝑞]_ [(] _[𝑙]_ [−][1][)] + e _[𝑐𝑞]_ _𝑖_ � _,_ (10)


where Wc and Wq are trainable weight matrix for concepts and
questions. We initialize the input of the SIEnc by the output of the
semantic information encoder:


x _[𝑞]_ _𝑖_ [(][0][)] = x _[𝑞]_ _𝑖_ _[,]_ [ x] _[𝑐]_ _𝑖_ [(][0][)] = x _[𝑐]_ _𝑖_ _[.]_ (11)


We use �qi ∈ R _[𝑑]_ and �ci ∈ R _[𝑑]_ to denote representation vectors of
the question _𝑞𝑖_ and the concept _𝑐𝑖_ after the _𝑘_ -layer graph encoder,
where _𝑑_, _𝑘_ are hyper-parameters.


**4.3** **Student State Encoder**


To effectively learn from highly sparse question-response data,

[12, 26, 27] suggest to transform the original question-response data
into concept-response data. Due to the fact that some questions
contain multiple concepts, SINKT averages representation vectors



e _[𝑐𝑞]_ =
_𝑖_ ∑︁

_𝑐_ _𝑗_ ∈N _𝑐𝑖_ _[𝑞]_



_𝛼𝑖,𝑗_ _[𝑐𝑞]_ [∗] �W _𝑐𝑞_ x _[𝑐]_ _𝑗_ � _,_ (4)


SINKT: A Structure-Aware Inductive Knowledge Tracing Model
with Large Language Model CIKM ’24, October 21–25, 2024, Boise, ID, USA.


Addition



**Concept-Question Graph**


**(a) Textual Information Encoder (TIEnc).**



**Graph Encode Layer 2**


... ...


**(b) Structural Information Encoder (SIEnc).**









|Concatenate|Col2|tenate|Col4|
|---|---|---|---|
|**Concatenate**||||
|**Concatenate**||||


|Col1|Col2|Col3|
|---|---|---|
||||
||~~**GRU**~~|~~** Cell**~~|


|Col1|or|Col3|
|---|---|---|
||||


**(c) Student State Encoder and Response Predictor.**



**Figure 3: The framework and details of SINKT.**



of concepts related to the question _𝑞𝑡_ to represent students’ learning
at time step _𝑡_



1
u _𝑡_ = |C _𝑞𝑡_ |



∑︁ c� _𝑖_ _._ (12)

_𝑐𝑖_ ∈C _𝑞𝑡_



where W _𝑝_ ∈ R [3] _[𝑑]_ and b _𝑝_ ∈ R are trainable parameters. To train all
parameters in SINKT, we choose the cross-entropy log loss between
the predicted response _𝑦𝑡_ +1 and the ground-truth response _𝑟𝑡_ +1 as
the objective function



To jointly represent the item and correctness of students’ response,
we introduce v _𝑡_ ∈ R [2] _[𝑑]_ as


u _𝑡_ ⊕ 0 _𝑟𝑡_ = 1
v _𝑡_ = � 0 ⊕ u _𝑡_ _𝑟𝑡_ = 0 _[,]_ (13)


where 0 ∈ R _[𝑑]_ is a zero-vector. To model learning history of students,
we use Gated Recurrent Unit (GRU) [6] to capture the sequential
learning behavior


u _𝑟_ = _𝜎_ (W _𝑟_ (v _𝑖_ ⊕ h _𝑡_ −1) + b _𝑟_ _,_ (14)


u _𝑧_ = _𝜎_ (W _𝑧_ (v _𝑖_ ⊕ h _𝑡_ ) + b _𝑧,_ (15)


u _ℎ_ = tanh(W _ℎ_ (v _𝑖_ ⊕(u _𝑟_ ∗ h _𝑡_ −1)) + b _ℎ_ ) _,_ (16)


h _𝑡_ = (1 − u _𝑧_ ) ∗ u _ℎ_ + u _𝑧_ ∗ h _𝑡_ −1 _._ (17)


Here _𝜎_ denotes the _sigmoid_ function, and W _𝑟_ _,_ b _𝑟_ _,_ W _𝑧,_ b _𝑧,_ W _ℎ,_ b _ℎ_ are
trainable parameters, where W _𝑟_ _,_ W _𝑧,_ W _ℎ_ ∈ R _[𝑑]_ [×][2] _[𝑑]_ and b _𝑟_ _,_ b _𝑧,_ b _ℎ_ ∈
R _[𝑑]_ _._ Hidden state h _𝑡_ ∈ R _[𝑑]_ represents the knowledge state of the student at time step _𝑡_ . The student state encoder considers the conceptlevel forgetting and knowledge acquisition pattern together during
student’s learning process.


**4.4** **Response Prediction**


When a student answers a question, his knowledge state of the
corresponding concepts and the description of the question would
influence his correctness. Given knowledge state of the student
ht at time step _𝑡_, the question representation vector �qt+1 and its
concept-level representation vector ut+1, we make the prediction
as follows


_𝑦𝑡_ +1 = _𝜎_ [�] Wp (h _𝑡_ ⊕ ˜q _𝑡_ +1 ⊕ u _𝑡_ +1) + b _𝑝_ � _._ (18)



L = −



_𝑇_
∑︁ ( _𝑟𝑡_ log _𝑦𝑡_ + (1 − _𝑟𝑡_ ) log(1 − _𝑦𝑡_ )) _._ (19)

_𝑡_ =1



**4.5** **Automated Pipeline for Inductive KT**


SINKT mainly focuses on the semantic and structural information,
which could be completely generated by LLMs. Meanwhile, SINKT
use textual representations instead of ID embeddings, which is
available even in the absence of learning history. Therefore, when
a new concept or a new question is introduced into the database of
ITS, SINKT is capable to capture the knowledge state of students
towards the new questions or concepts through automated processing. Figure 4 illustrates the automated ingestion process for _𝑐_ 4 and
_𝑞_ 5 in Figure 1. When a teacher adds a new concept to the ITS, LLMs
are invoked to integrate the new vertex into the Concept-Question
Graph. The teacher could continue to add a new question that
is related to this new concept, and the ITS further utilizes LLMs
to annotate the new question with concepts. The new conceptquestion heterogeneous graph is automatically constructed. When
it is necessary to assess a student’s mastery of the new question
_𝑞_ 5, SINKT can automatically perform text encoding and propagate
information through the graph, ultimately accomplishing response
prediction.
The automated ingestion pipeline enhances the ITS by providing
the flexibility to update and expand the database. This capability
not only streamlines the process of integrating new content but also
ensures that the system remains adaptive to evolving educational
needs and emerging topics. However, existing transductive KT
models are unable to achieve this.


CIKM ’24, October 21–25, 2024, Boise, ID, USA. Fu et al.



**Division**







~~**X**~~ **CannotPredict**



**Figure 4: An automated pipeline for the inductive KT task. In**
**the graph, a new concept** _𝑐_ 4 **and a new question** _𝑞_ 5 **are added**
**into the database.**


**5** **EXPERIMENT**


In this section, we conduct experiments on four real-world datasets
to evaluate the proposed SINKT framework. Specifically, we aim to
answer the following research questions:


**RQ1** How does the proposed SINKT performs compared to the
state-of-the-art KT models?

**RQ2** How does SINKT perform when a new question is added to
the ITS?

**RQ3** What is the influence of various componets of SINKT?
**RQ4** How do different hyperparameters affect the performance
of SINKT?

**RQ5** Is the concept relation graph generated by GPT-4 reasonable
and useful?


**5.1** **Datasets**


In our experiments, we evaluate our method on four datasets: (1) ASSIST09 [5], (2)ASSIST12 [6], (3) Junyi [7] and (4) Programming. ASSIST09,
ASSIST12, and Junyi are public datasets widely used in intelligent
education research. Programming is a private dataset with plain
text of questions and concepts. For each dataset, students answering
less than ten questions are removed. In order to simulate the situation of data sparsity, we choose 2,000 students’ learning history in
ASSIST12 and Junyi. The basic statistics of these four datasets are
listed in Table 1, and descriptions are as follows:


- **ASSIST09** is collected from an online tutor platform that teaches
students mathematics during the school year 2009 to 2010. The
dataset records the plain text of concepts and does not describe
questions.

- **ASSIST12** is also collected from the ASSIST platform from Sept
2012 to Oct 2013. Like ASSIST09, plain text of concepts is contained, but question descriptions are not provided.

- **Junyi** is collected between November 2010 and March 2015 from
the e-learning platform in Taiwan Junyi Academy. We choose
KC as the concept text without using question descriptions.

- **Programming** is collected from a commercial code-learning
platform from December 13, 2021 to February 17, 2023. The
dataset contains plain text of concepts and question descriptions.


5https://sites.google.com/site/assistmentsdata/home/2009-2010-assistmentdata?authuser=0
6https://sites.google.com/site/ASSISTdata/home/2012-13-school-data-with-affect
7https://pslcdatashop.web.cmu.edu/DatasetInfo?datasetId=1198



**Table 1: Statistics of all datasets. Q. and C. denote questions**
**and concepts. Edge density is the density of the concept cor-**
**relation graph generated by GPT-4.**

|Statistics ASSIST09 ASSIST12 Junyi Programming|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|Subjects<br>Students<br>Interactions<br>Questions<br>Concepts|Math<br>2,661<br>165,455<br>14,083<br>134|Math<br>2,000<br>159,545<br>31,229<br>240|Math<br>2,000<br>93,697<br>2,017<br>40|Programming<br>2,756<br>193,284<br>726<br>82|
|Q. per C.<br>Attempts per Q.<br>Edge Density|105.10<br>26.04<br>0.1067|130.12<br>7.70<br>0.0778|50.43<br>151.86<br>0.1750|17.89<br>603.50<br>0.0723|



**5.2** **Baselines**


To demonstrate the effectiveness of predicting students’ response,
we evaluate the performance of SINKT with 12 baselines.


- **DKT** [34] and **DHKT** [42] use deep learning methods on the KT
task, which uses RNN to model students’ learning history.

- **DKVMN** [50], **SKVMN** [3] apply key-value memory network
on the KT task.

- **CKT** [36] applies hierarchical convolutional layers and learns a
matrix demonstrating students’ mastery level of each concept.

- **SAKT** [32], **EERNNA** [37] and **AKT** [12] use attention mechanism to learn the past interactions’ importance in predicting
students’ current questions.

- **GKT** [30] generates a transition graph from the dataset and uses
GNN to encode the knowledge states of students.

- **SKT** [40] exploits similarity relation and prerequisites relation
between concepts based on GKT.

- **IEKT** [28] retrieves the similar histories of other students to
enhance the prediction of target students.

- **LBKT** [45] considers several dominant learning behaviors, including speed, attempts, and hints in the learning process.


**5.3** **Implementation Details**


In our experiment, we choose the most recent 200 records of students. We choose BERT to be the textual encoder for three public
datasets with concept text and choose Vicuna to be the textual
encoder for the Programming dataset with concept and question
text. We will compare the performance of different textual encoders
in Section 5.4. The layer number of the graph encoder is chosen
from {1 _,_ 2 _,_ 3}. The parameter _𝑑_ is set to 256. The learning rate is
chosen from {0 _._ 0001 _,_ 0 _._ 00005} with a decay at each epoch. The optimizer is Adam [16]. For a fair comparison, the hyper-parameters
of baselines are carefully chosen to have the best performance. Our
code is available now [8] .


**5.4** **Overall Performance (RQ1)**


We compare SINKT to 12 baselines on predicting the correctness of
students’ responses in four real-world datasets. The experimental
results are shown in Table 2. We use Accuracy (ACC) and Area
Under the Curve (AUC) as the evaluation metric.


8The PyTorch and MindSpore implementation is available at:
https://github.com/tubehao/SINKT.git and https://github.com/mindsporelab/models/tree/master/research/huawei-noah/SINKT.


SINKT: A Structure-Aware Inductive Knowledge Tracing Model
with Large Language Model CIKM ’24, October 21–25, 2024, Boise, ID, USA.


**Table 2: Overall performance of SINKT and baselines in four real-world datasets. SINKT-BERT and SINKT-Vicuna choose BERT**
**and Vicuna as TIEnc, respectively. Existing state-of-the-art results are underlined and the best results are bold. * indicates**
**p-value** _<_ 0 _._ 05 **in the t-test.**








|ASSIST09 ASSIST12 Junyi Programming<br>Groups Models<br>ACC AUC ACC AUC ACC AUC ACC AUC|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|**Groups**<br>**Models**<br>**ASSIST09**<br>**ASSIST12**<br>**Junyi**<br>**Programming**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**|**Groups**<br>**Models**<br>**ASSIST09**<br>**ASSIST12**<br>**Junyi**<br>**Programming**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**|**Groups**<br>**Models**<br>**ASSIST09**<br>**ASSIST12**<br>**Junyi**<br>**Programming**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**|**Groups**<br>**Models**<br>**ASSIST09**<br>**ASSIST12**<br>**Junyi**<br>**Programming**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**|**Groups**<br>**Models**<br>**ASSIST09**<br>**ASSIST12**<br>**Junyi**<br>**Programming**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**<br>**ACC**<br>**AUC**|**ACC**<br>**AUC**|
|ID Only|DKT<br>DHKT<br>DKVMN<br>SKVMN<br>CKT<br>SAKT<br>EERNNA<br>AKT<br>SKT|0.7166<br>0.7087<br>0.7357<br>0.7442<br>0.7175<br>0.7033<br>0.7201<br>0.7322<br>0.6672<br>0.6725<br>0.6509<br>0.6370<br>0.7247<br>0.7427<br>0.6683<br>0.6158<br>0.7082<br>0.7157|0.6936<br>0.6235<br>0.7038<br>0.6627<br>0.6975<br>0.6282<br>0.6952<br>0.6252<br>0.6492<br>0.5722<br>0.6626<br>0.5557<br>0.6964<br>0.6487<br>0.6652<br>0.5983<br>0.6858<br>0.6165|0.7516<br>0.7768<br><br>0.7563<br>0.7826<br><br>0.7625<br>0.7949<br><br>0.7675<br>0.8018<br><br>0.7268<br>0.7446<br><br>0.7381<br>0.7580<br><br>0.7558<br>0.7837<br><br>0.6864<br>0.6753<br><br>0.7307<br>0.7335<br>|0.7799<br>0.7957<br>0.7891<br>0.8130<br>0.7655<br>0.7651<br>0.7687<br>0.7736<br>0.7721<br>0.7815<br>0.7558<br>0.7680<br>0.7867<br>0.8071<br>0.7308<br>0.6251<br>0.7781<br>0.7898|
|Using Extra Info.|GKT<br>IEKT<br>LBKT|0.7370<br>0.7511<br>0.7031<br>0.6966<br>0.6819<br>0.6814|0.7160<br>0.6918<br>0.6880<br>0.6216<br>0.6747<br>0.6707|0.7345<br>0.7298<br><br>0.7168<br>0.7081<br><br>0.7456<br>0.7733<br>|0.7641<br>0.7507<br>0.7973<br>0.8261<br>0.7780<br>0.7898|
|Ours|SINKT-BERT<br>SINKT-Vicuna|**0.7416***<br>**0.7726***<br>0.7417<br>0.7700|**0.7264***<br>**0.7028***<br>0.7173<br>0.7003|**0.7721***<br>0.8069<br><br>0.7718<br>**0.8095***<br>**0**|0.7957<br>0.8225<br>**.7980***<br>**0.8267***|



In Table 2, several observations can be obtained as follows:


(1) SINKT significantly outperforms all comparative methods across
all datasets and evaluation metrics. This superior performance
indicates that SINKT provides more precise representations of
concepts and questions, as well as a more accurate estimation of
students’ knowledge states.
(2) When the datasets only contain textual information of concepts
(ASSIST09, ASSIST12, Junyi), using BERT [9] as the textual encoder yields better results. This is attributed to the fact that most
concept texts consist of 1-3 words, and models like BERT are
more adept at encoding word-level text. In contrast, when SINKT
is provided with question texts (Prgogramming), Vicuna [39]
demonstrates improved performance, since generative models
have a better understanding of sentence-level text typical of
question descriptions.
(3) Introducing additional specific information into the model does
not guarantee beneficial results across all datasets. For example,
GKT, which extracts a transition graph from training data to
train concept embeddings, shows significant performance improvement on the ASSIST09 and ASSIST12 datasets. However,
this approach introduces noise in the Junyi and Programming
datasets, detracting from model effectiveness.


**5.5** **Inductive Learning of SINKT (RQ2)**


To investigate the performance of SINKT after introducing new
questions, we remove 1/4 of the question from the training set and
evaluate the model’s prediction accuracy on these questions on the
validation and test sets. Since the other three datasets do not provide
textual information for the questions, we compare the performance
of SINKT with that of DHKT and EERNNA (best and second best
baselines) on the Programming dataset. We implement random
embedding initialization for DHKT and EERNNA on those unseen



**Table 3: Performance comparison of DHKT, EERNNA, and**
**SINKT in transductive and inductive KT tasks on Program-**
**ming dataset.**


|Models<br>KTTaskType Metrics<br>DHKT EERNNA SINKT|Col2|Col3|
|---|---|---|
|**KT Task Type**<br>**Metrics**<br>**Models**<br>**DHKT**<br>**EERNNA**<br>**SINKT**|**KT Task Type**<br>**Metrics**<br>**Models**<br>**DHKT**<br>**EERNNA**<br>**SINKT**|**DHKT**<br>**EERNNA**<br>**SINKT**|
|Transductive KT|ACC|0.7890<br>0.7867<br>0.7957|
|Transductive KT|AUC|0.8130<br>0.8071<br>0.8225|
|Inductive KT|ACC|0.5931<br>0.5781<br>0.6497|
|Inductive KT|AUC|0.5442<br>0.5502<br>0.6071|











**(a) Effect of training samples.**



|79<br>71<br>63|SINKT<br>EERNNA<br>DHKT|
|---|---|
|0<br>50<br>100<br>Training Steps<br>55|0<br>50<br>100<br>Training Steps<br>55|


**(b) Rate of convergence.**



**Figure 5: Cold start analysis of SINKT.**


questions to enable them to predict responses. The results of the
experiment are shown in the Table3. The experiment demonstrates
that the SINKT performs better in predicting unseen questions,
whereas the existing KT models struggle significantly with this
task, achieving an AUC of less than 0.6.
We also investigate the cold start problem in real ITS. We first
organize an experiment of different training samples on ASSIST09.
We choose {100 _,_ 500 _,_ 1000 _,_ 2000, 2661 (all of the dataset)} students’


CIKM ’24, October 21–25, 2024, Boise, ID, USA. Fu et al.


**Table 4: Ablation study of SINKT on all datasets.**


|ASSIST09 ASSIST12 Junyi Programming<br>Models<br>ACC AUC ACC AUC ACC AUC ACC AUC|Col2|Col3|Col4|Col5|
|---|---|---|---|---|
|SINKT|0.7416<br>0.7726<br>|0.7264<br>0.7028|0.7721*<br>0.8095|0.7980<br>0.826|
|SINKT-Linear<br>SINKT-GAT|0.7381<br>0.7556<br><br>0.7427<br>0.7680<br>|0.7130<br>0.6866<br>0.7116<br>0.6914|0.7366<br>0.7395<br>0.7692<br>0.8034|0.7755<br>0.784<br>0.7949<br>0.822|
|SINKT-Text<br>SINKT-Transition|0.7375<br>0.7706<br><br>0.7449<br>0.7709<br>|0.7141<br>0.7004<br>0.7106<br>0.7006|0.7666<br>0.8023<br>0.7700<br>0.8064|0.7947<br>0.819<br>0.7971<br>0.825|











**(d) Programming.**















**(a) ASSIST09.**



**(b) ASSIST12.**



**(c) Junyi.**



**(d) Programming.**



**Figure 6: Sensitivity analysis of layer number** _𝑘_ **on four datasets.**











**(b) ASSIST12.**



**(c) Junyi.**















**(a) ASSIST09.**



**Figure 7: Sensitivity analysis of representation dimension** _𝑑_ **on four datasets.**



learning history to train DHKT, EERNNA, and SINKT and evaluate
them on the full test set. As shown in Figure 5a, as the number of
students decreases, performance decline of SINKT is more gradual
compared to baselines, which indicates that SINKT can still learn
accurate concept/question representations using textual and structural information even with limited training data. Additionally, we
present the training curves of DHKT, EERNNA, and SINKT on the
ASSIST09 dataset in Figure 5b. We argue that SINKT converges
faster and exhibits minimal overfitting. This robustness in performance under limited data conditions underscores the effectiveness
of incorporating rich textual and structural information in SINKT.


**5.6** **Ablation Study (RQ3)**


To further investigate the importance of each module in SINKT, we
design three variants to conduct the ablation study, each of which
removes or changes one part from the original SINKT:


- **SINKT-Linear** removes the jumping knowledge module in the
graph encoder.

- **SINKT-GAT** removes GAT layers in the SIEnc, _i.e._ only uses a
linear layer to be a textual information adapter.




- **SINKT-Text** replaces the textual encoder with an embedding
layer, which learns representations of questions and concepts
from the dataset.

- **SINKT-Transition** replaces the concept generated by GPT-4
with the transition graph generated in the dataset.


Note that SINKT-Linear and SINKT-Graph still suitable for the
inductive KT task, while SINKT-Transition and SINKT-Text do not
support predicted responses to unseen questions.
We demonstrate the experimental results of SINKT and four
variants in Table 4. From the ablation study, we can derive several
conclusions regarding the impact of different components in SINKT.
Firstly, removing textual initialization (SINKT-Text) and the GAT
layer (SINKT-GAT) in SIEnc results in a performance decline, indicating that both structural and textual information contributes
to the learning of question and concept representations. Secondly,
removing the jumping knowledge layer leads to a significant performance drop. This is because our heterogeneous graph is unevenly
distributed, and the addition of the jumping knowledge layer allows the model to actively capture useful information. Thirdly, the


SINKT: A Structure-Aware Inductive Knowledge Tracing Model
with Large Language Model CIKM ’24, October 21–25, 2024, Boise, ID, USA.







|t|Col2|Col3|Col4|.<br>0.20<br>0.15<br>0.10<br>0.05<br>0.00|Col6|Col7|Concept|Col9|Col10|Col11|Col12|Col13|0<br>0<br>0<br>0<br>0|.<br>.8<br>.6<br>.4<br>.2<br>.0|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|Concep|


**(a) Transition graph.**




|t|Col2|
|---|---|
|Concep|Concep|



**(b) Concept graph generated by GPT-4.**







**(c) Concept graph with attention weights.**



**Figure 8: Concept graph comparison between different generation types.**






















|Col1|-|000<br>.|000<br>.|
|---|---|---|---|
||0.00|-|0.0062|
||0.00|0.0313<br>0.00|-|


|-|000<br>.|05877<br>.|
|---|---|---|
|0.1700|-|0.0391|
|0.1251|0.0313|-|



**Concept Graph with**
**Transition Graph** **Attention Score**



= Equivalent Fractions = Ordering Fractions


= Addition and Subtraction Fractions


**Figure 9: Detailed demonstration of transition graph and**
**concept graph with attention score.**


performance degradation of SINKT-Transition demonstrates the
effectiveness of the Concept Graph generated by GPT-4.


**5.7** **Parameter Sensitivity Analysis (RQ4)**


In order to have an insight into the graph information conduction,
we change the number of the Graph Encode layer _𝑘_ ranging from
0 to 3. Especially when _𝑘_ = 0, the graph encoder degrades into
a Linear layer without neighbor aggregation. The experimental
results on four datasets are shown in Figure 6. We find that SINKT
performs best in most cases when _𝑘_ = 2, which indicates that
second-order relationships in the concept-question heterogeneous
graph are crucial for the learning process. The primary second-order
relationships include concept-question-concept, concept-conceptquestion, and question-concept-question. In contrast, lower-order
relationships provide limited neighbor information, while higherorder relationships introduce noise.
Furthermore, we investigate the sensitivity of dimension _𝑑_ of
representations vectors ˜ _𝑞,_ ˜ _𝑐_ . We evaluate SINKT’s performance on
three different numbers of _𝑑_ : {128 _,_ 256 _,_ 512}. The experimental results on four datasets are shown in Figure 7. We suggest that the
best representation dimension of SINKT is 256. However, existing
transductive KT models usually choose 64 or 128 as the representation dimension of concepts and questions. The gap in the selection of dimension values primarily arises from the introduction of
open-world knowledge in SINKT. This knowledge is extensive and
requires additional dimensions to be effectively encoded.


**5.8** **Quality of the Graph Generation (RQ5)**


As we generate a concept relation graph by GPT-4, it is critical to
inspect the quality of the graph. To investigate the overall quality



of the concept-concept graph, we compare the transition graph
and our concept relation. Transition graph denotes the transition
probability of concept answering, which is derived from dataset
and could not be generated without training data. We demonstrates
the transition graph of ASSIST09 in Figure 8a. The weight in the
_𝑖_ -th row and the _𝑗_ -th column of the transition graph means the
probability of concept _𝑐𝑖_ appears after concept _𝑐_ _𝑗_ in the dataset.
Since the learning sequence of concepts are generated by ITS’s
recommendation algorithm, the distribution of prerequisite concepts in the transition graph is relatively fixed. The concept relation
graph generated by GPT-4 is shown in Figure 8b. The weight of _𝑐𝑖_
in the figure is |N1 _𝑐𝑖_ ~~_[𝑐]_~~ | [. Meanwhile, we visualize the attention score]

of the first GAT layer in Figure 8c. Compared to transition graph,
our concept graph with attention weight is more integrated.
We also give a case study for the concept graph’s rationality in
Figure 9. In the transition graph, concept _𝑒_ 1 _,𝑒_ 2 _,𝑒_ 3 nearly do not
have correlations. While in our concept graph, relationship between
every two concepts are clear and can be judged by reasonable

scores.


**6** **CONCLUSION**


In this paper, we introduce a structure-aware knowledge tracing
framework SINKT with large language model, which are suitable for
both transductive and inductive KT tasks. SINKT leverages LLMs
to build structural relationships between concepts and questions
and encode semantic information. To better aggregate these information, in the learning process of students, we carefully design a
textual information encoder, a structural information encoder and
a student state encoder. Experimental results on four real-world
datasets show that SINKT outperforms current ID-based KT models
on both transductive and inductive KT tasks. We also provide case
studies and ablation studies to prove the effectiveness of each module in SINKT. In the future, we will try to incorporate more wealthy
open-world knowledge and make more accurate predictions on
inductive KT tasks.


**ACKNOWLEDGMENTS**


The Shanghai Jiao Tong University team is partially supported by
National Natural Science Foundation of China (62177033, 62076161)
and Shanghai Municipal Science and Technology Major Project
(2021SHZDZX0102). The work is also sponsored by Huawei Innovation Research Program. We thank MindSpore [1] for the partial
support of this work.


CIKM ’24, October 21–25, 2024, Boise, ID, USA. Fu et al.



**REFERENCES**


[[1] 2020. MindSpore. https://www.mindspore.cn/](https://www.mindspore.cn/)

[2] 2022. Contrastive Learning for Knowledge Tracing. In _TheWebConf 2022_ .

[3] Ghodai Abdelrahman and Qing Wang. 2019. Knowledge Tracing with Sequential
Key-Value Memory Networks. In _Proceedings of the 42nd International ACM SIGIR_
_Conference on Research and Development in Information Retrieval (SIGIR ’19)_ . ACM.
[https://doi.org/10.1145/3331184.3331195](https://doi.org/10.1145/3331184.3331195)

[4] Arkadeep Acharya, Brijraj Singh, and Naoyuki Onoe. 2023. LLM Based Generation of Item-Description for Recommendation System. _Proceedings of the 17th_
_ACM Conference on Recommender Systems_ [(2023). https://api.semanticscholar.](https://api.semanticscholar.org/CorpusID:261823487)
[org/CorpusID:261823487](https://api.semanticscholar.org/CorpusID:261823487)

[5] Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. 2016. Neural Machine
[Translation by Jointly Learning to Align and Translate. arXiv:1409.0473 [cs.CL]](https://arxiv.org/abs/1409.0473)

[6] Junyoung Chung, Çaglar Gülçehre, KyungHyun Cho, and Yoshua Bengio. 2014.
Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling.
_CoRR_ [abs/1412.3555 (2014). arXiv:1412.3555 http://arxiv.org/abs/1412.3555](https://arxiv.org/abs/1412.3555)

[7] Albert T Corbett and John R Anderson. 1994. Knowledge tracing: Modeling
the acquisition of procedural knowledge. In _Proceedings of the 6th International_
_Conference on User Modeling_ . Springer, 32–41.

[8] Albert T Corbett and John R Anderson. 1994. Knowledge tracing: Modeling the
acquisition of procedural knowledge. _User modeling and user-adapted interaction_
4, 4 (1994), 253–278.

[9] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2018. BERT:
Pre-training of Deep Bidirectional Transformers for Language Understanding.
_arXiv preprint arXiv:1810.04805_ (2018).

[10] Yihao Fang, Xianzhi Li, Stephen W. Thomas, and Xiaodan Zhu. 2023. ChatGPT
as Data Augmentation for Compositional Generalization: A Case Study in Open
[Intent Detection. arXiv:2308.13517 [cs.CL]](https://arxiv.org/abs/2308.13517)

[11] A. Fatemi et al. 2023. Which Modality should I use - Text, Motif, or Image? :
Understanding Graphs with Large Language Models. _ar5iv_ (2023). Accessed:
2023-05-06.

[12] Aritra Ghosh, Neil Heffernan, and Andrew S. Lan. 2020. Context-Aware Attentive
[Knowledge Tracing. arXiv:2007.12324 [cs.LG]](https://arxiv.org/abs/2007.12324)

[13] Raia Hadsell, Sumit Chopra, and Yann LeCun. 2006. Dimensionality reduction
by learning an invariant mapping. In _2006 IEEE Computer Society Conference on_
_Computer Vision and Pattern Recognition (CVPR’06)_, Vol. 2. IEEE, 1735–1742.

[14] Chen hao, Xie Runfeng, Cui Xiangyang, Yan Zhou, Wang Xin, Xuan Zhanwei, and
Zhang Kai. 2023. LKPNR: LLM and KG for Personalized News Recommendation
[Framework. arXiv:2308.12028 [cs.IR]](https://arxiv.org/abs/2308.12028)

[15] Sepp Hochreiter and Jürgen Schmidhuber. 1997. Long short-term memory. _Neural_
_computation_ 9, 8 (1997), 1735–1780.

[16] Diederik P. Kingma and Jimmy Ba. 2014. Adam: A Method for Stochastic Optimization. _arXiv preprint arXiv:1412.6980_ (2014).

[17] Thomas N Kipf and Max Welling. 2017. Semi-supervised classification with graph
convolutional networks. In _5th International Conference on Learning Representa-_
_tions, ICLR 2017 - Conference Track Proceedings_ .

[18] Y. Kojima, X. Yao, et al. 2023. GPT4Graph: Can Large Language Models Understand Graph Structured Data? An Empirical Evaluation and Benchmarking. _ar5iv_
(2023). Accessed: 2023-05-06.

[19] Jinming Li, Wentao Zhang, Tian Wang, Guanglei Xiong, Alan Lu, and Gerard
Medioni. 2023. GPT4Rec: A Generative Framework for Personalized Recommen[dation and User Interests Interpretation. arXiv:2304.03879 [cs.IR]](https://arxiv.org/abs/2304.03879)

[20] Nan Li, Bo Kang, and Tijl De Bie. 2023. LLM4Jobs: Unsupervised occupation extraction and standardization leveraging Large Language Models.
[arXiv:2309.09708 [cs.CL]](https://arxiv.org/abs/2309.09708)

[21] Pan Li, Yuyan Wang, Ed H. Chi, and Minmin Chen. 2023. Prompt Tuning
Large Language Models on Personalized Aspect Extraction for Recommendations.
[arXiv:2306.01475 [cs.IR]](https://arxiv.org/abs/2306.01475)

[22] Jianghao Lin, Xinyi Dai, Yunjia Xi, Weiwen Liu, Bo Chen, Hao Zhang, Yong Liu,
Chuhan Wu, Xiangyang Li, Chenxu Zhu, Huifeng Guo, Yong Yu, Ruiming Tang,
and Weinan Zhang. 2023. How Can Recommender Systems Benefit from Large
Language Models: A Survey. _arXiv preprint arXiv:2306.05817_ (2023).

[23] Jianghao Lin, Rong Shan, Chenxu Zhu, Kounianhua Du, Bo Chen, Shigang Quan,
Ruiming Tang, Yong Yu, and Weinan Zhang. 2024. ReLLa: Retrieval-enhanced
Large Language Models for Lifelong Sequential Behavior Comprehension in
Recommendation. In _Proceedings of the ACM on Web Conference 2024 (WWW ’24)_ .
3497–3508.

[24] Qijiong Liu, Nuo Chen, Tetsuya Sakai, and Xiao-Ming Wu. 2023. A First Look at
LLM-Powered Generative News Recommendation. _ArXiv_ abs/2305.06566 (2023).
[https://api.semanticscholar.org/CorpusID:263891105](https://api.semanticscholar.org/CorpusID:263891105)

[25] Qijiong Liu, Nuo Chen, Tetsuya Sakai, and Xiao-Ming Wu. 2023. ONCE: Boosting
Content-based Recommendation with Both Open- and Closed-source Large
[Language Models. arXiv:2305.06566 [cs.IR]](https://arxiv.org/abs/2305.06566)

[26] Zitao Liu, Qiongqiong Liu, Jiahao Chen, Shuyan Huang, and Weiqi Luo.
2023. simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Trac[ing. arXiv:2302.06881 [cs.LG]](https://arxiv.org/abs/2302.06881)




[27] Zitao Liu, Qiongqiong Liu, Jiahao Chen, Shuyan Huang, Jiliang Tang, and Weiqi
Luo. 2023. pyKT: A Python Library to Benchmark Deep Learning based Knowl[edge Tracing Models. arXiv:2206.11460 [cs.LG]](https://arxiv.org/abs/2206.11460)

[28] Ting Long, Yunfei Liu, Jian Shen, Weinan Zhang, and Yong Yu. 2021. Tracing Knowledge State with Individual Cognition and Acquisition Estimation. In
_Proceedings of the 44th International ACM SIGIR Conference on Research and_
_Development in Information Retrieval_ . ACM, 173–182.

[29] Sheshera Mysore, Andrew McCallum, and Hamed Zamani. 2023. Large Language
[Model Augmented Narrative Driven Recommendations. arXiv:2306.02250 [cs.IR]](https://arxiv.org/abs/2306.02250)

[30] Hiromi Nakagawa, Yusuke Iwasawa, and Yutaka Matsuo. 2019. Graph-based
Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network. In _Proceedings of the 2019 IEEE/WIC/ACM International Conference on Web_
_Intelligence (WI)_ [. https://doi.org/10.1145/3350546.3352513](https://doi.org/10.1145/3350546.3352513)

[[31] OpenAI. 2023. Introducing GPT-4. https://openai.com/blog/gpt-4.](https://openai.com/blog/gpt-4)

[32] Shalini Pandey and George Karypis. 2019. A Self-Attentive model for Knowledge
Tracing. In _Proceedings of the 12th International Conference on Educational Data_
_Mining (EDM 2019)_ . International Educational Data Mining Society.

[33] Wenjun Peng, Derong Xu, Tong Xu, Jianjin Zhang, and Enhong Chen. 2023. Are
GPT Embeddings Useful for&nbsp;Ads and&nbsp;Recommendation?. In _Knowl-_
_edge Science, Engineering and Management: 16th International Conference, KSEM_
_2023, Guangzhou, China, August 16–18, 2023, Proceedings, Part IV_ (Guangzhou,
[China). Springer-Verlag, Berlin, Heidelberg, 151–162. https://doi.org/10.1007/978-](https://doi.org/10.1007/978-3-031-40292-0_13)
[3-031-40292-0_13](https://doi.org/10.1007/978-3-031-40292-0_13)

[34] Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami,
Leonidas J. Guibas, and Jascha Sohl-Dickstein. 2015. Deep knowledge tracing. In
_Advances in Neural Information Processing Systems_, Vol. 28. 505–513.

[35] David E Rumelhart, Geoffrey E Hinton, and Ronald J Williams. 1986. Learning
representations by back-propagating errors. _Nature_ 323, 6088 (1986), 533–536.

[36] Shuanghong Shen, Qi Liu, Enhong Chen, Han Wu, Zhenya Huang, Weihao Zhao,
Yu Su, Haiping Ma, and Shijin Wang. 2020. Convolutional Knowledge Tracing:
Modeling Individualization in Student Learning Process. In _Proceedings of the 43rd_
_International ACM SIGIR Conference on Research and Development in Information_
_Retrieval_ (Virtual Event, China) _(SIGIR ’20)_ . Association for Computing Machinery,
[New York, NY, USA, 1857–1860. https://doi.org/10.1145/3397271.3401288](https://doi.org/10.1145/3397271.3401288)

[37] Yu Su, Qingwen Liu, Qi Liu, Zhenya Huang, Yu Yin, Enhong Chen, Chris H. Q.
Ding, Si Wei, and Guoping Hu. 2018. Exercise-Enhanced Sequential Modeling
for Student Performance Prediction. In _AAAI_ . 2435–2443.

[38] Zhaoxuan Tan and Meng Jiang. 2023. User Modeling in the Era of Large Language
[Models: Current Research and Future Directions. arXiv:2312.11518 [cs.CL]](https://arxiv.org/abs/2312.11518)

[39] Vicuna Team. 2023. Vicuna: An Open-Source Chatbot Impressing GPT-4 with
[90%* ChatGPT Quality. https://huggingface.co/lmsys/vicuna-13b-v1.5.](https://huggingface.co/lmsys/vicuna-13b-v1.5) _Hugging_
_Face_ (2023).

[40] Shiwei Tong, Qi Liu, Wei Huang, Zhenya Hunag, Enhong Chen, Chuanren Liu,
Haiping Ma, and Shijin Wang. 2020. Structure-Based Knowledge Tracing: An
Influence Propagation View. In _2020 IEEE International Conference on Data Mining_
_(ICDM)_ [. 541–550. https://doi.org/10.1109/ICDM50108.2020.00063](https://doi.org/10.1109/ICDM50108.2020.00063)

[41] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All
You Need. In _Advances in neural information processing systems_ . 5998–6008.

[42] T. Wang, F. Ma, and J. Gao. 2019. Deep Hierarchical Knowledge Tracing. In
_Proceedings of the 12th International Conference on Educational Data Mining_ .
667–670.

[43] Jason Weston, Sumit Chopra, and Antoine Bordes. 2015. Memory Networks.
[arXiv:1410.3916 [cs.AI]](https://arxiv.org/abs/1410.3916)

[44] Yunjia Xi, Weiwen Liu, Jianghao Lin, Xiaoling Cai, Hong Zhu, Jieming Zhu,
Bo Chen, Ruiming Tang, Weinan Zhang, Rui Zhang, and Yong Yu. 2023. Towards Open-World Recommendation with Knowledge Augmentation from Large
[Language Models. arXiv:2306.10933 [cs.IR]](https://arxiv.org/abs/2306.10933)

[45] Bihan Xu, Zhenya Huang, Jiayu Liu, Shuanghong Shen, Qi Liu, Enhong Chen,
Jinze Wu, and Shijin Wang. 2023. Learning Behavior-oriented Knowledge Tracing.
In _Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and_
_Data Mining_ (<conf-loc>, <city>Long Beach</city>, <state>CA</state>, <country>USA</country>, </conf-loc>) _(KDD ’23)_ . Association for Computing Machin[ery, New York, NY, USA, 2789–2800. https://doi.org/10.1145/3580305.3599407](https://doi.org/10.1145/3580305.3599407)

[46] Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi
Kawarabayashi, and Stefanie Jegelka. 2018. Representation Learning on
Graphs with Jumping Knowledge Networks. _CoRR_ abs/1806.03536 (2018).
[arXiv:1806.03536 http://arxiv.org/abs/1806.03536](https://arxiv.org/abs/1806.03536)

[47] Fan Yang, Zheng Chen, Ziyan Jiang, Eunah Cho, Xiaojiang Huang, and Yanbin Lu. 2023. PALR: Personalization Aware LLMs for Recommendation.
[arXiv:2305.07622 [cs.IR]](https://arxiv.org/abs/2305.07622)

[48] Liang Yang, Binbin Cui, Chengjie Wu, Chao Wang, Xing Zhang, and Jian Liu.
2020. GIKT: A Graph-based Interaction Model for Knowledge Tracing. _arXiv_
_preprint arXiv:2002.07033_ (2020).

[49] Chun-Kit Yeung and Dit-Yan Yeung. 2018. Addressing Two Problems in Deep
Knowledge Tracing via Prediction-Consistent Regularization. In _Proceedings of_
_the Fifth Annual Conference on Learning at Scale_ (London, UK), Rose Luckin, Scott
Klemmer, and Kenneth R. Koedinger (Eds.). ACM, 5:1–5:10.


SINKT: A Structure-Aware Inductive Knowledge Tracing Model
with Large Language Model CIKM ’24, October 21–25, 2024, Boise, ID, USA.




[50] Jiani Zhang, Xingjian Shi, Irwin King, and Dit-Yan Yeung. 2017. Dynamic Key[Value Memory Networks for Knowledge Tracing. arXiv:1611.08108 [cs.AI]](https://arxiv.org/abs/1611.08108)

[51] Moyu Zhang, Xinning Zhu, Chunhong Zhang, Yang Ji, Feng Pan, and Changchuan
Yin. 2021. Multi-Factors Aware Dual-Attentional Knowledge Tracing. In _Proceed-_
_ings of the 30th ACM International Conference on Information &amp; Knowledge_
_Management (CIKM ’21)_ [. ACM. https://doi.org/10.1145/3459637.3482372](https://doi.org/10.1145/3459637.3482372)

[52] Moyu Zhang, Xinning Zhu, Chunhong Zhang, Wenchen Qian, Feng Pan, and
Hui Zhao. 2023. Counterfactual Monotonic Knowledge Tracing for Assessing



Students’ Dynamic Mastery of Knowledge Concepts. In _Proceedings of the 32nd_
_ACM International Conference on Information and Knowledge Management (CIKM_
_’23)_ [. ACM. https://doi.org/10.1145/3583780.3614827](https://doi.org/10.1145/3583780.3614827)

[53] Zhi Zheng, Zhaopeng Qiu, Xiao Hu, Likang Wu, Hengshu Zhu, and Hui
Xiong. 2023. Generative Job Recommendations with Large Language Model.
[arXiv:2307.02157 [cs.IR]](https://arxiv.org/abs/2307.02157)


