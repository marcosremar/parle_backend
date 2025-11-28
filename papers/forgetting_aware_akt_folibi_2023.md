## **Forgetting-aware Linear Bias for Attentive Knowledge Tracing**



Yoonjin Im [∗]

Sungkyunkwan University
Suwon, Republic of Korea
skewondr@skku.edu


Heejin Kook
Sungkyunkwan University
Suwon, Republic of Korea
hjkook@skku.edu


**ABSTRACT**


Knowledge Tracing (KT) aims to track proficiency based on a
question-solving history, allowing us to offer a streamlined curriculum. Recent studies actively utilize attention-based mechanisms
to capture the correlation between questions and combine it with
the learner’s characteristics for responses. However, our empirical
study shows that existing attention-based KT models neglect the
learner’s _forgetting behavior_, especially as the interaction history
becomes longer. This problem arises from the bias that overprioritizes the correlation of questions while inadvertently ignoring the
impact of forgetting behavior. This paper proposes a simple-yeteffective solution, namely _**Fo**_ _rgetting-aware_ _**Li**_ _near_ _**Bi**_ _as (FoLiBi)_, to
reflect forgetting behavior as a linear bias. Despite its simplicity,
FoLiBi is readily equipped with existing attentive KT models by
effectively decomposing question correlations with forgetting behavior. FoLiBi plugged with several KT models yields a consistent
improvement of up to 2.58% in AUC over state-of-the-art KT models
on four benchmark datasets.


**CCS CONCEPTS**


- **Social and professional topics** → **Student assessment** ; • **Com-**
**puting methodologies** → **Neural networks** .


**KEYWORDS**


Knowledge tracing, Attention mechanism, Forgetting behavior


**ACM Reference Format:**

Yoonjin Im, Eunseong Choi, Heejin Kook, and Jongwuk Lee. 2023. Forgettingaware Linear Bias for Attentive Knowledge Tracing. In _Proceedings of the_
_32nd ACM International Conference on Information and Knowledge Manage-_
_ment (CIKM ’23), October 21–25, 2023, Birmingham, United Kingdom._ ACM,
[New York, NY, USA, 5 pages. https://doi.org/10.1145/3583780.3615191](https://doi.org/10.1145/3583780.3615191)


∗Equal contribution

- Corresponding author


Permission to make digital or hard copies of all or part of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for components of this work owned by others than the
author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specific permission
and/or a fee. Request permissions from permissions@acm.org.
_CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom._
© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM.
ACM ISBN 979-8-4007-0124-5/23/10...$15.00
[https://doi.org/10.1145/3583780.3615191](https://doi.org/10.1145/3583780.3615191)



Eunseong Choi [∗]

Sungkyunkwan University
Suwon, Republic of Korea
eunseong@skku.edu


Jongwuk Lee [†]

Sungkyunkwan University
Suwon, Republic of Korea
jongwuklee@skku.edu


**1** **INTRODUCTION**


With the prevalence of various online education platforms and
intelligent tutoring systems, analyzing the learner’s proficiency
through her interactions is crucial. _Knowledge Tracing (KT)_ [4] aims
to quantify the level of knowledge by tracking the learner’s responses to the preceding questions and to predict the performance
of new questions. To effectively solve this task as a sequential prediction, it is essential to learn the correlation between questions
( _e.g._, question similarity and concept dependency) and the characteristics of the learner’s response ( _e.g._, forgetting behavior and
concept proficiency).
Recent KT models utilized various deep neural networks to capture the question correlation: (i) Deep sequential models represent
learner’s knowledge with hidden states of RNNs [6, 12, 13, 15, 17,
23, 29, 35]. (ii) Memory-augmented models [2, 36] use key-value
memory structures to update knowledge. (iii) Graph-based models
employ GNNs to capture the correlation between questions and
concepts [19, 30, 34]. (iv) Attentive models that apply several variants of attention mechanisms to reflect long-term relationships of
questions [8, 14, 16, 20, 27].
Besides, several studies [3, 7, 11, 18, 21, 26, 28] attempted to consider _forgetting behavior_ based on a learning curve theory [1] into
deep learning models to integrate the learner’s latent factors. A
common assumption is that the learners’ knowledge is increasingly
forgotten over time. DFKT [18] and LPKT [26] extend DKT [23]
by leveraging time-interval information to reflect the forgetting
effect of learners. HawkesKT [33] utilizes time interval information
to improve an existing matrix-factorization-based KT model [32].
SAINT+ [28] extends SAINT [8] by using the time gap between
questions from different concept sets. However, these methods
need help establishing discrete representations for continuous time
intervals, mainly when they encounter time outliers that exceed
the threshold of pre-defined embeddings. AKT [11] and RKT [21]
address this challenge by introducing a relative time interval bias
that penalizes positional decay in attention weights instead of using absolute position embeddings. Nonetheless, the bias becomes
unnecessarily entangled with question correlations, obscuring the
influence of forgetting behavior. It is found that this tendency is
more evident in longer interactions, suggesting the disentanglement between question correlations and forgetting behavior.
Inspired by recent work [24], we propose a more fine-designed
method to model forgetting behavior built upon attention-based
KT models, _**Fo**_ _rgetting-aware_ _**Li**_ _near_ _**Bi**_ _as (FoLiBi)_, decoupling the


CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom. Yoonjin Im, Eunseong Choi, Heejin Kook, & Jongwuk Lee


|Col1|Table 1: Dataset statistics.|
|---|---|
|Dataset|# learners<br># interactions<br># concepts<br># questions<br>% correct|
|AL05<br>BR06<br>AS09<br>SLPY|571<br>607,014<br>271<br>173,113<br>76<br>1,138<br>1,817,450<br>550<br>129,263<br>78<br>3,695<br>282,071<br>149<br>17,728<br>62<br>5,000<br>623,212<br>1,380<br>2,713<br>79|



_𝜶_ _𝑡_ = softmax **q** _𝑡_ **K** ⊤
� ~~√~~ _𝑑_



_,_ **v** _𝑡_ = _𝜶_ _𝑡_ **V** _,_ (1)
�



~~√~~



_𝑑_



where _𝑑_ is the dimension size of key, query, and value embedding
vectors. Let **q** _𝑡_ ∈ R [1][×] _[𝑑]_, **K** ∈ R [(] _[𝑡]_ [−][1][)×] _[𝑑]_, and **V** ∈ R [(] _[𝑡]_ [−][1][)×] _[𝑑]_ denote
the query vector for the target question _𝑒𝑡_, the key matrix for
previous questions (1 ≤ _𝑖_ _< 𝑡_ ), and the value matrix for previous
responses, respectively. The scaled dot product [31] captures the
relevance of each previous question with respect to the current
questions. **v** _𝑡_ is computed as a weighted sum of the values.


**Modeling forgetting behavior** . The correlation between a previous and the target question diminishes over time, indicating a
gradual decay of the user’s memory. We integrate the concept of
forgetting behavior in AKT [11] and RKT [21] into the attention
blocks presented, resulting in a simplified formula.



**Figure 1: The architecture of the forgetting-aware attentive**
**KT model. We describe the proposed method in the knowl-**
**edge retriever, a typical framework for attentive KT models.**


learner’s forgetting effect from question correlations. It can be
achieved through linear bias, which penalizes the question attention
scores proportional to the relative distance. Despite its simplicity,
it has two advantages. (i) It does not interfere with estimating the
correlations of questions. (ii) It shows the robustness against the various sequence lengths. We have validated that FoLiBi consistently
outperforms existing methods for modeling forgetting behavior on
four standard datasets when plugged into four different KT models.


**2** **PROPOSED METHOD**


**Problem definition** . Let _𝑋𝑡_ = ⟨ _𝑥_ 1 _,𝑥_ 2 _...,𝑥𝑡_ ⟩ denote a sequence
of interactions at the _𝑡_ -th time step. Each interaction _𝑥𝑖_ = ( _𝑒𝑖,𝑟𝑖_ )
consists of a pair of a question _𝑒𝑖_ and its correctness _𝑟𝑖_ ∈{0 _,_ 1}. If _𝑒𝑖_
is answered correctly, _𝑟𝑖_ = 1, otherwise _𝑟𝑖_ = 0. Given the sequence
of interactions _𝑋𝑡_ −1, KT models aim to predict the probability of
correctly answering the question _𝑒𝑡_, _i.e._, ^ _𝑟𝑡_ = _𝑝_ ( _𝑟𝑡_ = 1 | _𝑋𝑡_ −1 _,𝑒𝑡_ ).
They commonly adopt a binary cross-entropy for a loss function.


**Attentive knowledge tracing models** . Figure 1 shows the architecture of attention-based KT models based on the previous
works [11, 14]. Given sequences of questions ⟨ _𝑒_ 1 _,𝑒_ 2 _, ...,𝑒𝑡_ ⟩ and interactions _𝑋𝑡_ −1, we get the knowledge vector of the learner _𝒐𝒕_ ∈ R [1][×] _[𝑑]_

by passing through three attention mechanism-based structures:
question encoder, interaction encoder, and knowledge retriever.
It then combines _𝒐𝒕_ with the target question’s embedding vector
**h** _𝑡_ ∈ R [1][×] _[𝑑]_, and feeds it to a classification layer consisting of twolayered MLP with a sigmoid function to get a final prediction ^ _𝑟𝑡_ .
While the two encoders and the knowledge retriever have similar
internal architectures, our primary focus lies on the knowledge
retriever. It calculates the question correlations as attention weights
and aggregates interactions as a linear combination.



where _𝜷𝑡_ ∈ R [1][×(] _[𝑡]_ [−][1][)] and _𝜸𝑡_ ∈ R [1][×(] _[𝑡]_ [−][1][)] are additional coefficients
to represent forgetting behavior according to _𝑡_ -th time step. By
combining these two additional terms into the attention mechanism,
this approach considers both question correlations and forgetting
behavior in a unified manner.
AKT [11] uses _monotonic attention (Mono)_, which incorporates
forgetting behavior into the attention weight. Monotonic attention
can generally be expressed with coefficient _𝜷𝑡_ as:


_𝜷𝑡_ [Mono] = (exp(− _𝜃ℎ_    - **d** _𝑡_ ) − 1) · **q** _𝑡_ **K** [⊤] _, 𝜸𝑡_ [Mono] = 0 [�] _,_ (3)


where _𝜃ℎ_ is a learnable decay rate parameter for the _ℎ_ -th head
within the multi-head attention block. **d** _𝑡_ = [ _𝑑_ (1 _,𝑡_ ) _,𝑑_ (2 _,𝑡_ ) _, . . .,𝑑_ ( _𝑡_ −
1 _,𝑡_ )] ∈ R [1][×(] _[𝑡]_ [−][1][)] consists of the function _𝑑_ ( _𝑖,𝑡_ ), which calculates
the time interval bias among time steps _𝑖_ and _𝑡_ for a given question by multiplying the question similarity with relative distance.
In other words, **d** _𝑡_ penalizes positional decay but unnecessarily
intertwines with the correlation of the question.
Meanwhile, RKT [21] develops a _relational coefficient (RC)_ that
considers question similarity and forgetting behavior. It can be
defined with coefficient _𝜸𝑡_ as:


_𝜷𝑡_ [RC] = 0 [�] _, 𝜸𝑡_ [RC] = softmax( **R** _[𝐸]_ + **R** _[𝑇]_ ) _,_ (4)


where **R** _[𝐸]_ ∈ R [1][×(] _[𝑡]_ [−][1][)] represents the similarities between previous
questions and the target question derived from textual embeddings,
and **R** _[𝑇]_ ∈ R [1][×(] _[𝑡]_ [−][1][)] is defined as [exp(−( _𝑡_ − 1)/ _𝑆_ ) _,_ exp(−( _𝑡_ −
2)/ _𝑆_ ) _, . . .,_ exp(−1/ _𝑆_ )], consisting of the relative time interval normalized by a learnable decay rate _𝑆_ . Because **R** _[𝐸]_ denotes the similarity of the context between the questions, the question correlation
is again considered in the attention operation eventually, which
reduces the effect of the relative time interval.
Despite diligent efforts, Eq. (3) and (4) are insufficient in capturing the exclusive influence of relative positional decay, resulting in



_𝜶_ _𝑡_ = softmax **q** _𝑡_ **K** ⊤ + _𝜷𝑡_
� ~~√~~ _𝑑_



+ _𝜸𝑡_ _,_ (2)
�


Forgetting-aware Linear Bias for Attentive Knowledge Tracing CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom.


**Table 2: Overall performance comparisons. The best scores are in bold, and the second-best scores are** **underlined. Statistical**
**significant differences (** _𝑝_ _<_ 0 _._ 05 **) between the reproduced baseline models and FoLiBi are reported with** [⋄] **. The last column**
**shows the average ratio of the improvement over the original method,** _i.e._ **, PE for SAKT [20] and Mono for AKT [11], CL4KT [14].**






|Model Type|AL05<br>AUC ACC RMSE|BR06<br>AUC ACC RMSE|AS09<br>AUC ACC RMSE|SLPY<br>AUC ACC RMSE|AUC improv.<br>/Original|
|---|---|---|---|---|---|
|**SAKT**<br>PE<br>Mono<br>RC<br>FoLiBi|75.69~~⋄~~<br>78.50~~⋄~~<br>39.37~~⋄~~<br>75.62⋄<br>78.57⋄<br>39.20⋄<br>76.31⋄<br>78.68⋄<br>39.09⋄<br>**77.96**<br>**79.40**<br>**38.57**|74.37~~⋄~~<br>78.83~~⋄~~<br>39.13~~⋄~~<br>74.43⋄<br>79.01⋄<br>38.77⋄<br>75.42⋄<br>79.15⋄<br>38.71⋄<br>**76.72**<br>**79.69**<br>**38.25**|73.81~~⋄~~<br>70.60~~⋄~~<br>44.42~~⋄~~<br>73.62⋄<br>70.38⋄<br>44.35⋄<br>74.12⋄<br>70.90⋄<br>44.23⋄<br>**75.24**<br>**71.78**<br>**43.68**|69.02~~⋄~~<br>77.94<br>40.11<br>69.35⋄<br>**78.21**<br>**39.64**<br>69.62⋄<br>77.35<br>40.18<br>**70.55**<br>77.52<br>40.01|2.58%|
|**AKT**<br>PE<br>Mono<br>RC<br>FoLiBi|74.13~~⋄~~<br>77.68~~⋄~~<br>39.88~~⋄~~<br>76.73⋄<br>78.61<br>38.93⋄<br>76.63⋄<br>78.62<br>38.89⋄<br>**77.64**<br>**78.89**<br>**38.64**|74.64~~⋄~~<br>78.47~~⋄~~<br>38.86~~⋄~~<br>76.18⋄<br>79.31<br>38.26⋄<br>76.42⋄<br>79.25<br>38.22⋄<br>**77.09**<br>**79.39**<br>**38.09**|73.49~~⋄~~<br>70.27~~⋄~~<br>44.59~~⋄~~<br>74.91⋄<br>71.25⋄<br>43.80<br>74.81⋄<br>71.13⋄<br>43.92⋄<br>**75.28**<br>**71.60**<br>**43.63**|70.48~~⋄~~<br>78.06<br>39.66~~⋄~~<br>72.42⋄<br>78.49<br>39.03<br>72.15⋄<br>**78.71**<br>39.13⋄<br>**72.78**<br>78.69<br>**38.90**|0.84%|
|**CL4KT**<br>PE<br>Mono<br>RC<br>FoLiBi|74.93~~⋄~~<br>77.76~~⋄~~<br>39.64~~⋄~~<br>77.72⋄<br>79.09⋄<br>38.52⋄<br>77.03⋄<br>78.94⋄<br>38.71⋄<br>**78.97**<br>**79.69**<br>**38.09**|74.02~~⋄~~<br>78.39~~⋄~~<br>39.16~~⋄~~<br>76.65⋄<br>79.55⋄<br>38.10⋄<br>76.58⋄<br>79.70<br>38.08⋄<br>**77.80**<br>**80.00**<br>**37.73**|74.00~~⋄~~<br>70.48~~⋄~~<br>44.39~~⋄~~<br>75.47⋄<br>71.54⋄<br>43.56⋄<br>75.35⋄<br>71.36⋄<br>43.70⋄<br>**76.20**<br>**72.00**<br>**43.29**|68.99~~⋄~~<br>78.18<br>40.01~~⋄~~<br>71.27<br>78.57<br>39.25<br>71.23<br>**78.71**<br>39.23<br>**71.59**<br>78.69<br>**39.19**|1.13%|
|**CL4KT**<br>w/o CL<br>PE<br>Mono<br>RC<br>FoLiBi|75.68~~⋄~~<br>78.16~~⋄~~<br>39.30~~⋄~~<br>77.72⋄<br>79.25⋄<br>38.45⋄<br>77.54⋄<br>79.24⋄<br>38.49⋄<br>**79.24**<br>**79.89**<br>**37.89**|75.78~~⋄~~<br>79.16~~⋄~~<br>38.54~~⋄~~<br>76.96⋄<br>79.68⋄<br>37.94⋄<br>77.31⋄<br>79.99⋄<br>37.78⋄<br>**78.23**<br>**80.22**<br>**37.53**|74.31~~⋄~~<br>70.73~~⋄~~<br>44.23~~⋄~~<br>75.18⋄<br>71.35⋄<br>43.72⋄<br>75.49⋄<br>71.57⋄<br>43.57⋄<br>**76.17**<br>**72.07**<br>**43.32**|71.04~~⋄~~<br>78.25~~⋄~~<br>39.56~~⋄~~<br>72.02⋄<br>78.75<br>39.14⋄<br>72.15⋄<br>78.76<br>38.97⋄<br>**72.90**<br>**78.80**<br>**38.76**|1.54%|



only modest enhancements (see Section 3.2 for further discussion).
We analyzed that both methods do not account for position decay
in a balanced manner due to unnecessary redundancy.
The linear bias [24] computes a constant value based on relative
distance and provides an effective approach to penalize forgetting
behavior directly. This paper devises a simple-yet-effective solution,
called _forgetting-aware linear bias (FoLiBi)_, to decouple the intricate
relationship between question correlation and forgetting behavior.
We represent forgetting behavior using coefficient _𝜷𝑡_ as:


_𝜷𝑡_ [FoLiBi] = _𝑚ℎ_       - [1 _, . . .,𝑡_ − 1] _, 𝜸𝑡_ [FoLiBi] = 0 [�] _,_ (5)


−8
where [1 _, . . .,𝑡_ − 1] is a linear bias over time and _𝑚ℎ_ = 2 _𝐻_ [·] _[ℎ]_

denotes a coefficient that adjusts the importance of forgetting behavior for the _ℎ_ -th head out of _𝐻_ attention heads.

Figure 2 shows how the attention weights of the existing methods and FoLiBi are distributed according to the positions. Because
FoLiBi independently computes the question correlations, it effectively penalizes positional decay for forgetting behavior, especially
as the length of history increases, _e.g._, 20. Our experiments have
shown that this change improves overall performance, suggesting
that FoLiBi not only penalizes position collapse but also balances
correlations between questions at the same time.


**3** **EXPERIMENTS**

**3.1** **Experimental Setup**


**Datasets.** We evaluate the effectiveness of the proposed method using four real-world educational datasets: Algebra 2005-2006 (AL05),
Bridge to Algebra 2006-2007 (BR06), ASSISTments 2009-2010 (AS09)

[9], and Slepemapy (SLPY) [22]. AL05 and BR06 are algebra learning
history datasets provided by KDD Cup 2010. AS09 is collected from
a free online math education platform. SLPY, a dataset dedicated
to geography learning history, originates from the online platform
_slepemapy.cz_, and we randomly sampled 5,000 learners out of 91,331.



For all datasets, we follow the same pre-processing procedure in

[10, 14], such as dropping learners whose number of interactions
is less than 5. Table 1 shows statistics of the processed datasets.


**Baselines and evaluation metrics.** We apply FoLiBi and other
forgetting behavior modeling methods to several transformer-based
models to analyze their effectiveness thoroughly. SAKT [20] is an
early model of knowledge tracing using an attention mechanism.
AKT [11] introduces embeddings based on the Rasch model [25]
and reflects the learner’s forgetting behavior in the attention mechanism. CL4KT [14] proposes contrastive learning on the augmented
learning histories. We only report the method used in RKT [21],
_i.e._, RC, because it integrates textual content into the input in the
backbone model. Following the previous works [14, 26], we adopt
Area Under Curve (AUC), Accuracy (ACC), and Root Mean Squared
Error (RMSE), which are commonly used in KT for evaluation.


**Implementation details.** We perform 5-fold cross-validation and
report the average of 3 different seeds. For validation, we split 10%
from each training fold. Following [11, 14], we regard the same
concepts as a single question to help avoid over-parameterization.
We set the maximum history length to 100 for all experiments
except the one for figure 3. The hyperparameters are optimized for
each dataset using Optuna [5]. All the source code is available. [1]


**3.2** **Results and Analysis**

To verify the effectiveness of our proposed method, we address the
following research questions.


 - **RQ1** Does FoLiBi improve the performance of the state-of-theat KT models?

 - **RQ2** What are the main differences between FoLiBi and existing
methods on attention weights?

 - **RQ3** Does FoLiBi perform better on longer histories?


[1https://github.com/skewondr/FoLiBi](https://github.com/skewondr/FoLiBi)


CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom. Yoonjin Im, Eunseong Choi, Heejin Kook, & Jongwuk Lee



**FoLiBi**


1 10 20


1 10 20



**RC**


1 10 20


1 10 20



0.78


0.76


0.74


0.74


0.72


0.7


0.68



2


11


21


2


11


21



**Mono**


1 10 20


1 10 20



1


0.8


0.6


0.4


0.2


0


1


0.8


0.6


0.4


0.2


0



**BR06**


10 20 50 100 200 300


**FoLiBi**

|1|10 2|20 5|50 10 SLPY|00 20|Col6|00 30|
|---|---|---|---|---|---|---|
||||||||
|||||||**PE**<br>**Mono**<br>**RC**<br>|
||||||||
||||||||



10 20 50 100 200 300



**Past position index**


**Figure 2: Visualization of attention weights with different**
**methods for forgetting behavior on the AKT. Note that the**
**value in the upper triangle is always 0 due to the causal mask.**


**Overall performance comparison** . Table 2 shows the performance of different forgetting behavior modeling methods using
different backbone models on the four benchmark datasets. While
Positional Embedding (PE) incorporates absolute position information into the input embedding, Monotonic attention (Mono),
Relational Coefficient (RC), and FoLiBi design the coefficients _𝜷𝒕_
and _𝜸𝒕_ to reflect relative positional decay. The key observations are:
(i) All backbone models show the highest AUC performance when
FoLiBi is plugged in. It confirms that FoLiBi helps trace learners’
knowledge effectively by avoiding the intervention between two
objectives, _i.e._, question and positional correlation. (ii) Mono and RC
usually perform better than PE by accounting for relative distance.
However, in most cases, they have no significant improvement because both approaches neglect the negative effects of the question
correlation when considering forgetting behavior. (iii) Applying
FoLiBi to CL4KT [14] shows an improvement over the original
method, Mono, with a 1.13% in AUC on average across datasets,
but removing contrastive learning (CL) increases the improvement
to 1.54%. As the contrastive learning in CL4KT [14] leads the model
to be more dependent on the question relevance, it implies that the
balance between the question and position correlation is crucial.


**Comparison via attention weights.** Figure 2 visualizes the attention weights of the first 20 records in the first and last head of
the final self-attention block in AKT [11]. The x and y-axis of each
map represent the previous and the target questions, respectively.
Mono and RC models tend to assign a weight spread across all
previous questions in the first head. On the other hand, we found
that FoLiBi holds the more relative position for rearward sequences.
This is because Mono and RC are overly affected by question correlation when computing _𝜷𝑡_ and _𝜸𝑡_ . Nevertheless, FoLiBi uses a
non-learnable _𝜷𝑡_, which means that a positional correlation can
constantly influence the attention weight regardless of the model
performance. While the other two methods maintain the tendency
within heads, FoLiBi adjusts the importance of position correlation
with the slope _𝑚ℎ_, so that the last head assigns attention weights
based mainly on the question correlations. This allows FoLiBi to
capture complex interactions, resulting in consistent effectiveness
over varying lengths. We will discuss this in the next section.



**AL05**


0.8


0.78


0.76


0.74

10 20 50 100 200 300


**AS09**
0.76


0.75


0.74


0.73


0.72


10 20 50 100 200 300



**History length**


**Figure 3: Performance comparison varying the number of**
**previous interactions used as input for inference. We used**
**the AKT model as the backbone and trained with a maximum**

**length of 300 for a fair comparison to the PE method.**


**Effectiveness on various history lengths.** Assuming the importance of forgetting behavior depends on the length of the problemsolving history, we conducted further experiments to see how effective each method is in varying lengths. We set the maximum
sequence length to 300 in training and fixed the length of input
sequence _𝑛_, _i.e._, 10, 20, 50, 100, 200, or 300, in evaluation. For example, if _𝐿_ is the total length of a learner’s history, then the number
of questions evaluated for the learner is _𝐿_ − _𝑛_ . We make the following observations. Firstly, the lines of RC, Mono, and FoLiBi are
higher than those of PE, and the gaps between them get larger as
the number of interactions increases. This suggests that relative
positional distance is a core factor in the KT task and becomes more
critical with a longer history. Second, the improvement of FoLiBi is
maintained or increased as the number of interactions increases.
It indicates that FoLiBi can reflect forgetting behavior better than
other methods owing to decoupling from question correlations.


**4** **CONCLUSION**


This paper first analyzes the effect of forgetting behavior in existing
attention-based KT models. As the interaction sequence extends
over time, the correlation between questions becomes disproportionately emphasized compared to the impact of forgetting behavior. To address this problem, we propose FoLiBi, which represents
forgetting behavior as a linear bias decoupled from the question
correlation. Despite its simplicity, extensive experiments show that
FoLiBi consistently outperforms previous KT models and distinguishes the relative position distance from the question correlation.
As a result, FoLiBi is suitable to account for forgetting-aware attention by readily plugging into existing attention-based KT models.


**ACKNOWLEDGMENTS**


This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant and National
Research Foundation of Korea (NRF) grant funded by the Korea
government (MSIT) (No. 2019-0-00421, 2022-0-00680, 2022-0-01045,
2021-0-01560, and NRF-2018R1A5A1060031).


Forgetting-aware Linear Bias for Attentive Knowledge Tracing CIKM ’23, October 21–25, 2023, Birmingham, United Kingdom.



**REFERENCES**


[1] Hermann Ebbinghaus (1885). 2013. Memory: A Contribution to Experimental
Psychology. _Annals of Neurosciences_ 20 (2013), 155.

[2] Ghodai Abdelrahman and Qing Wang. 2019. Knowledge Tracing with Sequential
Key-Value Memory Networks. In _SIGIR_ . 175–184.

[3] Ghodai Abdelrahman and Qing Wang. 2021. Deep Graph Memory Networks for
Forgetting-Robust Knowledge Tracing. _CoRR_ abs/2108.08105 (2021).

[4] Ghodai Abdelrahman, Qing Wang, and Bernardo Pereira Nunes. 2022. Knowledge
Tracing: A Survey. _CoRR_ [abs/2201.06953 (2022). https://arxiv.org/abs/2201.06953](https://arxiv.org/abs/2201.06953)

[5] Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori
Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework. In _SIGKDD_ . 2623–2631.

[6] Penghe Chen, Yu Lu, Vincent W. Zheng, and Yang Pian. 2018. Prerequisite-Driven
Deep Knowledge Tracing. In _ICDM_ . 39–48.

[7] Yuying Chen, Qi Liu, Zhenya Huang, Le Wu, Enhong Chen, Run-ze Wu, Yu
Su, and Guoping Hu. 2017. Tracking Knowledge Proficiency of Students with
Educational Priors. In _CIKM_ . 989–998.

[8] Youngduck Choi, Youngnam Lee, Junghyun Cho, Jineon Baek, Byungsoo Kim,
Yeongmin Cha, Dongmin Shin, Chan Bae, and Jaewe Heo. 2020. Towards an
Appropriate Query, Key, and Value Computation for Knowledge Tracing. In _L@S_ .
341–344.

[9] Mingyu Feng, Neil T. Heffernan, and Kenneth R. Koedinger. 2009. Addressing
the assessment challenge with an online system that tutors as it assesses. _User_
_Model. User Adapt. Interact._ 19 (2009), 243–266.

[10] Theophile Gervet, Ken Koedinger, Jeff Schneider, and Tom Mitchell. 2020. When is
Deep Learning the Best Approach to Knowledge Tracing? _Journal of Educational_
_Data Mining_ 12 (2020), 31–54.

[11] Aritra Ghosh, Neil T. Heffernan, and Andrew S. Lan. 2020. Context-Aware
Attentive Knowledge Tracing. In _SIGKDD_ . 2330–2339.

[12] Xiaopeng Guo, Zhijie Huang, Jie Gao, Mingyu Shang, Maojing Shu, and Jun Sun.
2021. Enhancing Knowledge Tracing via Adversarial Training. In _MM_ . 367–375.

[13] Jinseok Lee and Dit-Yan Yeung. 2019. Knowledge Query Network for Knowledge
Tracing: How Knowledge Interacts with Skills. In _LAK_ . 491–500.

[14] Wonsung Lee, Jaeyoon Chun, Youngmin Lee, Kyoungsoo Park, and Sungrae Park.
2022. Contrastive Learning for Knowledge Tracing. In _WWW_ . 2330–2338.

[15] Qi Liu, Zhenya Huang, Yu Yin, Enhong Chen, Hui Xiong, Yu Su, and Guoping
Hu. 2021. EKT: Exercise-Aware Knowledge Tracing for Student Performance
Prediction. _TKDE_ 33 (2021), 100–115.

[16] Ting Long, Jiarui Qin, Jian Shen, Weinan Zhang, Wei Xia, Ruiming Tang, Xiuqiang He, and Yong Yu. 2022. Improving Knowledge Tracing with Collaborative
Information. In _WSDM_ . 599–607.

[17] Sein Minn, Yi Yu, Michel C. Desmarais, Feida Zhu, and Jill-Jênn Vie. 2018. Deep
Knowledge Tracing and Dynamic Student Classification for Knowledge Tracing.
_ICDM_ (2018), 1182–1187.

[18] Koki Nagatani, Qian Zhang, Masahiro Sato, Yan-Ying Chen, Francine Chen,
and Tomoko Ohkuma. 2019. Augmenting Knowledge Tracing by Considering



Forgetting Behavior. In _WWW_ . 3101–3107.

[19] Hiromi Nakagawa, Yusuke Iwasawa, and Yutaka Matsuo. 2019. Graph-based
Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network.
In _WI_ . 156–163.

[20] Shalini Pandey and George Karypis. 2019. A Self Attentive model for Knowledge
Tracing. In _EDM_ .

[21] Shalini Pandey and Jaideep Srivastava. 2020. RKT: Relation-Aware Self-Attention
for Knowledge Tracing. In _CIKM_ . 1205–1214.

[22] Jan Papoušek, Radek Pelánek, and Vít Stanislav. 2016. Adaptive Geography
Practice Data Set. _Journal of Learning Analytics_ 3 (2016), 317–321.

[23] Chris Piech, Jonathan Bassen, Jonathan Huang, Surya Ganguli, Mehran Sahami,
Leonidas J. Guibas, and Jascha Sohl-Dickstein. 2015. Deep Knowledge Tracing.
In _NIPS_ . 505–513.

[24] Ofir Press, Noah A. Smith, and Mike Lewis. 2022. Train Short, Test Long: Attention
with Linear Biases Enables Input Length Extrapolation. In _ICLR_ .

[25] Georg Rasch. 1993. Probabilistic Models for Some Intelligence and Attainment
Tests. In _MESA Press_ .

[26] Shuanghong Shen, Qi Liu, Enhong Chen, Zhenya Huang, Wei Huang, Yu Yin, Yu
Su, and Shijin Wang. 2021. Learning Process-consistent Knowledge Tracing. In
_SIGKDD_ . 1452–1460.

[27] Shuanghong Shen, Qi Liu, Enhong Chen, Han Wu, Zhenya Huang, Weihao Zhao,
Yu Su, Haiping Ma, and Shijin Wang. 2020. Convolutional Knowledge Tracing:
Modeling Individualization in Student Learning Process. In _SIGIR_ . 1857–1860.

[28] Dongmin Shin, Yugeun Shim, Hangyeol Yu, Seewoo Lee, Byungsoo Kim, and
Youngduck Choi. 2021. SAINT+: Integrating Temporal Features for EdNet Correctness Prediction. In _LAK_ . 490–496.

[29] Yu Su, Qingwen Liu, Qi Liu, Zhenya Huang, Yu Yin, Enhong Chen, Chris Ding, Si
Wei, and Guoping Hu. 2018. Exercise-Enhanced Sequential Modeling for Student
Performance Prediction. _AAAI_ 32 (2018).

[30] Shiwei Tong, Qi Liu, Wei Huang, Zhenya Hunag, Enhong Chen, Chuanren Liu,
Haiping Ma, and Shijin Wang. 2020. Structure-Based Knowledge Tracing: An
Influence Propagation View. In _ICDM_ . 541–550.

[31] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All
you Need. In _NIPS_ . 5998–6008.

[32] Jill-Jênn Vie and Hisashi Kashima. 2019. Knowledge Tracing Machines: Factorization Machines for Knowledge Tracing. In _AAAI_ . 750–757.

[33] Chenyang Wang, Weizhi Ma, Min Zhang, Chuancheng Lv, Fengyuan Wan, Huijie
Lin, Taoran Tang, Yiqun Liu, and Shaoping Ma. 2021. Temporal Cross-Effects in
Knowledge Tracing. In _WSDM_ . 517–525.

[34] Yang Yang, Jian Shen, Yanru Qu, Yunfei Liu, Kerong Wang, Yaoming Zhu, Weinan
Zhang, and Yong Yu. 2021. GIKT: A Graph-Based Interaction Model for Knowledge Tracing. In _ECML PKDD_ . 299–315.

[35] Chun-Kit Yeung and Dit-Yan Yeung. 2018. Addressing Two Problems in Deep
Knowledge Tracing via Prediction-Consistent Regularization. In _L@S_ .

[36] Jiani Zhang, Xingjian Shi, Irwin King, and Dit-Yan Yeung. 2017. Dynamic KeyValue Memory Networks for Knowledge Tracing. In _WWW_ . 765–774.


