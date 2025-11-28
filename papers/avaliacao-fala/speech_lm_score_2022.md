**SPEECHLMSCORE: EVALUATING SPEECH GENERATION USING**

**SPEECH LANGUAGE MODEL**


_Soumi Maiti_ [1] _, Yifan Peng_ [1] _, Takaaki Saeki_ [1] _[,]_ [2] _, Shinji Watanabe_ [1]



1Carnegie Mellon University, USA, 2The University of Tokyo, Japan



**ABSTRACT**


While human evaluation is the most reliable metric for evaluat
ing speech generation systems, it is generally costly and timeconsuming. Previous studies on automatic speech quality assessment address the problem by predicting human evaluation scores
with machine learning models. However, they rely on supervised
learning and thus suffer from high annotation costs and domain-shift
problems. We propose _SpeechLMScore_, an unsupervised metric to evaluate generated speech using a speech language model.
SpeechLMScore computes the average log-probability of a speech
signal by mapping it into discrete tokens and measures the average
probability of generating the sequence of tokens. Therefore, it does
not require human annotation and is a highly scalable framework.
Evaluation results demonstrate that the proposed metric shows a
promising correlation with human evaluation scores on different
speech generation tasks including voice conversion, text-to-speech,
and speech enhancement.


_**Index Terms**_ **—** automatic speech quality assessment, speech
language model, discrete token


**1. INTRODUCTION**











Human evaluation is the gold standard for evaluating different
speech generation systems including speech synthesis, voice conversion, and speech enhancement. It is typically carried out with
subjective listening tests such as Mean-Opinion-Score (MOS) tests
on 1-5 Likert scale, where a participant listens to an audio sample
and provides judgement. Such subjective tests are time-consuming,
costly and hence are not scalable. In contrast, objective evaluation
metrics are easy to use for large numbers of systems and can provide
faster results but these objective metrics are generally not highly correlated with the human judgement score. For example text-to-speech
(TTS) objective metrics like Mean-Cepstral-Distortion (MCD) [1]
or F0-root mean square error (F0-RMSE) poorly correlate with human evaluation as they only measure certain acoustic similarity with
a reference speech. Similar phenomena are observed for speech
enhancement objective scores like PESQ [2].
To tackle this problem, recent studies [3–6] treat the speech evaluation as a regression or classification task and train a neural network
to predict MOS. Such models require large-scale listening tests with
multiple participants for supervision. For example, recent work [4]
conducts listening test with 1.3M samples for multilingual MOS, [7]
collected human evaluation on 7K samples. In addition to the high
cost of collecting human judgements, such supervised models also
risks poor generalization to new tasks and domains. In this paper, we
propose to evaluate speech generation by measuring the likelihood
with a speech language model, instead of predicting non-absolute
human scores. In recent years, speech language models [8–10] are
introduced for the speech continuation task or unconditional natural



**Fig. 1** . Speech Language Model [8] using discrete speech units.


speech generation. Here, we argue that we can also use such speech
language models for measuring likelihood of another speech sample.
More specifically, this paper proposes an unsupervised speech
evaluation metric, _SpeechLMScore_ based on speech language models. SpeechLMScore measures the likelihood of a speech sample
to natural speech using a pretrained speech-unit language model.
SpeechLMScore is non-intrusive i.e., it does not require the reference speech for measuring similarities unlike some objective
measures MCD, F0-RMSE or PESQ. More importantly, SpeechLMScore is an unsupervised metric, hence it does not need explicit
human MOS labels. SpeechLMScore is not optimized for a particular task and hence is more generalizable than other metrics.
We show that SpeechLMScore is highly correlated with different
speech evaluation tasks like voice conversion, TTS, and speech enhancement. Especially, SpeechLMScore shows higher correlation
in mismatched domains compared to supervised MOS prediction
models. Furthermore, we show that using duration information of
speech tokens in speech language model, correlation is improved.
We will open-source SpeechLMScore evaluation metric, speech language model training code and pretrained speech language model
using ESPnet toolkit [1] .


**2. RELATED WORK**



In the field of natural language generation, unsupervised automatic
evaluation methods with language models [11–13] have been proposed. Unlike traditional metrics such as BLEU [14], these evaluation methods can capture more semantic information inherent in the
language model, leading to higher robustness and correlation with
human evaluations. In recent years, some studies perform audio
generation as neural machine translation [15] or language modeling
tasks [8,10] by treating acoustic signals as discrete tokens, opening
up a new frontier in audio generation. Therefore, our study naturally
extends the language-model-based scoring to a speech generation
task, which is a promising and highly scalable framework.
Automatic speech quality assessment methods for speech generation tasks such as speech synthesis [3,7,16] and speech enhancement [17] are actively being studied. Recently, transfer learning
from self/semi-supervised representation learning models [4, 6, 18]
has shown remarkable prediction accuracy and generalization abil

[1https://github.com/ESPnet/ESPnet](https://github.com/ESPnet/ESPnet)


(a)



**Table 1** . Utterance and system-level correlation with MOS in VoiceMOS challenge 2022 dataset (7106 files) [7] with different configurations: layer number ( _L_ ) to extract feature from Hubert and number
of clusters ( _V_ ). We use SpeechLMScore with pretrained uLM.



For _i_ [th] token























(b)


**Fig. 2** . Computation of SpeechLMScore: (a) Given speech waveform **x**, encoder generates discrete tokens (b) For each token _di_, a
unit language model generates probability matrix over vocabulary _V_
given all previous tokens. Finally, SpeechLMScore( **x** ) is calculated
as the average log-likelihood of all tokens.


ity. However, these models are trained with supervised learning and
thus require large-scale listening tests to collect a large number of
target labels. Therefore, previous work aims to learn MOS prediction models with non-matching references for the application to outof-domain data [19]. In contrast to the above studies, our approach
performs scoring with a speech language model and allows automatic evaluations without any human annotations.


**3. METHOD**


Given a _t_ -length speech signal **x** = _{x_ 1 _· · · xt}_, _xt ∈_ R, we can
compute a joint probability of speech signal using chain rule as


_p_ ( **x** ) = Π _i_ _[t]_ =1 _[p]_ [(] _[x]_ _i_ _[|][x]_ _<i_ [)] _[,]_ (1)


where _p_ ( _xi|x<i_ ) = _p_ ( _xi|x_ 1 _· · · xi−_ 1). A speech language model _θ_,
generates the probability distribution of speech **x** as


_p_ ( **x** _|θ_ ) = Π _i_ _[t]_ =1 _[p]_ [(] _[x]_ _i_ _[|][x]_ _<i_ _[, θ]_ [)] _[.]_ (2)


We propose to estimate quality of speech **x** with average logprobability of speech computed using a speech language model.


**3.1. Speech language model**


A speech language model was proposed as the task of learning linguistic and acoustic features of speech. Such models generate probability distribution of speech samples using pseudo-text units [8,
10, 15]. A speech language model generally consists of two main
submodules: i) _tokenizer_ (Tok) and ii) autoregressive _unit language_
_model_ (uLM) as shown in Fig. 1.
A tokenizer takes continuous speech signal **x** as input and maps
into a series of discrete units or tokens **d** as


Tok( **x** ) = **d** = _{d_ 1 _· · · dT },_ (3)


where each _di ∈{_ 1 _· · · V }_, _V_ is the size of vocabulary of such
discrete tokens and _T_ is the total number of discrete tokens in **x**,
_T < t_ . In general, it is similar to a tokenizer in a text language


|Utterance-level System-level<br>ID V L LCC SRCC KTAU LCC SRCC KTAU|Col2|Col3|Col4|
|---|---|---|---|
|50~~ 3~~<br>50~~ 4~~<br>50~~ 6~~<br>50~~ 1~~2|50<br>3<br>4<br>6<br>12|**0.472**<br>0.490<br>0.343<br>0.464<br>0.492<br>0.344<br>0.462<br>0.462<br>0.321<br>0.279<br>0.348<br>0.234|0.753<br>0.749<br>0.549<br>**0.760**<br>**0.755**<br>**0.562**<br>0.694<br>0.692<br>0.496<br>0.514<br>0.555<br>0.388|
|100~~ 2~~<br>100~~ 3~~<br>100~~ 4~~<br>100~~ 5~~<br>100~~ 6~~<br>100~~ 1~~2|100<br>2<br>3<br>4<br>5<br>6<br>12|0.376<br>0.460<br>0.321<br>0.322<br>0.505<br>0.355<br>0.379<br>0.527<br>0.370<br>0.282<br>0.482<br>0.337<br>0.300<br>0.454<br>0.317<br>0.289<br>0.375<br>0.259|0.673<br>0.683<br>0.503<br>0.598<br>0.666<br>0.490<br>0.705<br>0.741<br>0.552<br>0.552<br>0.615<br>0.444<br>0.523<br>0.559<br>0.392<br>0.532<br>0.562<br>0.394|
|200~~ 3~~<br>200~~ 4~~<br>200~~ 6~~|3<br>200<br>4<br>6|0.419<br>**0.538**<br>**0.380**<br>0.464<br>0.536<br>0.378<br>0.360<br>0.487<br>0.342|0.719<br>0.726<br>0.539<br>0.701<br>0.700<br>0.511<br>0.594<br>0.649<br>0.471|



model. Tokenizer consists of an encoder (generally self-supervisedlearning models) that extracts continuous features from speech **x** :
**f** = _{f_ 1 _, f_ 2 _· · · fT }_, where _fi ∈_ R _[D]_ and _D_ is the dimension of
encoded feature. A quantizer then maps **f** into discrete speech unit
tokens **d** . For example GSLM [8] uses CPC [20], HuBERT [21]
and wav2vec 2.0 [22] as feature encoder followed by k-means clustering [23] as quantizer. Here the number of clusters ( _k_ ) used in
k-means clustering defines the unit vocabulary _V_ .
A speech unit language model is an autoregressive neural network that models the probability distribution over the set of discrete
tokens **d** as
_p_ ( **d** ) = Π _i_ _[T]_ =1 _[p]_ [(] _[d]_ _i_ _[|][d]_ _<i_ [)] _[.]_ (4)


A uLM is trained to maximize the log-likelihood on a large speech
dataset. Ideally, we can use any autoregressive network for the
speech language model. Speech unit language models can generate human-like speech or continue a speech when fragment of
speech is provided as context, similar to text generation capability
of text-based language models as shown in [8].


**Speech discrete units for GSLM:** In this work, we use Generative Spoken Language Modelling (GSLM) as the pretrained speech
language model. GSLM uses small number of tokens: _V_ = 50 _,_ 100
and 200. Discrete units are generated from speech at a rate of
20 ms. Stacks of Transformer [24] decoder modules are used
for language model network. Before unit language model training, GSLM removes repeated discrete tokens, i.e., the sequence
20 _,_ 20 _,_ 20 _,_ 16 _,_ 17 _,_ 17 converts to 20 _,_ 16 _,_ 17. The authors in [8]
hypothesize that such removal helps with limited attention span of
Transformer network. Though such compression can help with reducing sequence length, it also looses useful information of speech
token duration. Hence, we train a new language model to retain
such information. We use the same Tokenizer as GSLM and train a

new uLM without removal of repeated tokens. Since such repeated
tokens increases the sequence length, we use a LSTM [25] language
model for less memory requirement.


**3.2. SpeechLMScore**


As noted in Section 3.1, uLM is trained to maximize the loglikelihood on a speech-only dataset. Thus a robust speech language
model can be used to estimate the likelihood of a given speech.




**Table 2** . Utterance and System-level correlation with MOS in VoiceMOS 2022 challenge testset (1066 files) and whole dataset (7106 files)
with SpeechLMScore.


**test-set** **whole-set**

**Utterance-level** **System-level** **Utterance-level** **System-level**
Model LCC SRCC KTAU LCC SRCC KTAU LCC SRCC KTAU LCC SRCC KTAU


Matched training domain


MOSNnet (pre) 0.454 0.480 0.339 0.481 0.459 0.323 0.415 0.432 0.302 0.518 0.497 0.356
MOSNet (ft) **0.868** **0.865** **0.690** **0.948** **0.944** **0.803**      -      -      -      -      -      

Mismatched training domain


DNSMOS (SIG) **0.536** **0.553** **0.392** **0.652** **0.684** **0.498** **0.495** **0.503** **0.354** **0.714** **0.720** **0.532**
DNSMOS (BAK) 0.266 0.298 0.204 0.370 0.410 0.282 0.293 0.317 0.219 0.429 0.410 0.280
DNSMOS (OVRL) 0.496 0.497 0.352 0.606 0.623 0.450 0.473 0.473 0.334 0.678 0.668 0.488


Unsupervised


SpeechLMScore (Pre) 0.452 0.524 0.371 0.711 0.745 0.547 0.490 0.472 0.343 0.749 **0.754** 0.549
SpeechLMScore (LSTM) 0.538 0.539 0.383 0.720 0.728 0.531 0.497 0.499 0.350 0.753 0.748 0.554
SpeechLMScore (LSTM)+rep 0.582 0.572 0.410 **0.743** **0.749** **0.551** **0.519** **0.516** **0.367** **0.759** 0.739 **0.564**
SpeechLMScore (LSTM) (Large) 0.540 0.536 0.381 0.709 0.724 0.529 0.496 0.497 0.349 0.745 0.744 0.551
SpeechLMScore (LSTM)+rep (Large) **0.586** **0.584** **0.419** 0.729 0.736 0.539 0.514 **0.516** 0.365 0.749 0.733 0.542



Here we propose that such a speech likelihood can also be used to
evaluate the quality of speech. That is, given speech language model
_θ_, SpeechLMScore( **x** _|θ_ ) is defined as



The testset contains 150 noisy samples, with varying signal-to-noise
ratio (SNR) from 0-25 dB. Synthetic noisy speech contains clean
speech from [34], and 12 noise categories from AudioSet [35]: fan,
air conditioner, typing, door shutting, clatter noise, car, munching,
creaking chair, breathing, copy machine, baby crying, and barking.


**Speech unit language models:** As discussed in section 3.1, We
used pretrained models from GSLM [8] using fairseq [2] . We use a
pretrained tokenizer using HUBERT-BASE-LS960H as the encoder
and k-means clustering models as the quantizer. HUBERT-BASELS960H consist of 12 Transformer encoder layers and trained with
LibriSpeech 960 hour dataset [30]. There are three versions of
pretrained clustering models: _V_ = 50 _,_ 100 and 200 trained on LibriSpeech clean-100 hour subset. We use Hubert [21] features and
the corresponding clustering model as that was best performing
uLM in the GSLM. Pretrained uLM consists of 12 Transformer

decoder layers and trained using 3072 maximum sequence length
and on a “clean” subset containing 6K hours speech [20] selected
from LibriLight 60K dataset [36]. For our language model training,
we use a LSTM language model. We train using two datasets: i)
LibriLight medium segmented set with 5.6K hours of speech, and
ii) another large dataset of approximately three times larger (16.8K
hours) randomly selected from the LibriLight 60K hour dataset. We
use a smaller subset from LibriLight 60K dataset for less computation requirement. Our LSTM language model contains 1024 hidden
units with 3 layers and is trained with 0.2 dropout, 0.002 learning
rate with Adam [37] optimizer for 40 epochs. For the larger dataset
model (Large), we use a similar configuration with 4 layers. In this
work, we report these speechLMscore models: SpeechLMScore
(Pre): uses the GSLM pretrained model, SpeechLMScore (LSTM):
uses our LSTM language model with removal of repeated tokens
as GSLM, SpeechLMScore(LSTM)+rep: uses our trained LSTM
model with repeated tokens, and SpeechLMScore(Large): language
models trained on the larger subset.


**Supervised baselines:** We compare our unsupervised speechLMscore with two supervised MOS prediction models: MOSNet [5]
and DNSMOS [5]. MOSNet is trained on the speech synthesis domain. The MOSNet pretrained model, MOSNet (Pre), is trained


[2https://github.com/facebookresearch/fairseq](https://github.com/facebookresearch/fairseq)



SpeechLMScore( **d** _|θ_ ) = [1]

_T_



_T_
� log _p_ ( _di|d<i, θ_ ) _._ (5)


_i_ =1



More specifically, SpeechLMScore using uLM can be computed as
follows: i) encode the speech into discrete tokens **d** = _{d_ 1 _· · · dT }_,
and ii) for all tokens iteratively compute, log probability of current
token _di_ provided all previous tokens _{d_ 1 _· · · di−_ 1 _}_ using _θ_, i.e.,
log _p_ ( _di|d<i, θ_ ) as shown in Fig. 2.
SpeechLMScore measures the average log-probability of a set
of speech tokens. Such metric is also related to perplexity of a
speech sample. In fact, it is measuring how perplexed a speech language model is given set of discrete tokens from speech **x** . Thus,
lower perplexity, or higher log-likelihood, should correlate with human evaluations of higher speech quality. In the next section, we
present experimental results demonstrating the effectiveness of the
SpeechLMScore metric.


**4. EXPERIMENTS**


In this section, we evaluate the generalizability of the proposed unsupervised metric through different speech evaluation tasks.


**4.1. Experimental setup**


**Dataset:** Currently, the largest dataset to compare with human
scores in speech synthesis/voice conversion is the VoiceMOS challenge [7]. The main track of the challenge dataset contains a total of
7,106 utterances with 187 different systems and 38 samples per system. The dataset contains speech from three main sources: i) voice
conversion challenges (VCC) from 2016, 2018 and 2020, a total of
79 systems [26–28], ii) more recent text-to-speech models trained
with ESPnet [29] on LJspeech [30], 10 systems, and iii) older speech
synthesis systems from past Blizzard challenges [31–33] (20082016), a total of 98 systems. For each utterance, 8 human judgments
are collected and the average score is used as ground truth human
judgement. Next, we evaluate our approach on noisy speech and
measure the correlation with the noisiness of speech using the Deep
Noise Suppression challenge 2020 [17] noisy reverberant testset.


**Table 3** . Utterance-level and System-level correlation with MOS
scores in different sources of VoiceMOS 2022 challenge dataset.
VCC contains 3002 utterances, ES-TTS 380 and BLZ 3724.

|Source|uLM|Utterance-level<br>LCC SRCC KTAU|System-level<br>LCC SRCC KTAU|
|---|---|---|---|
|VCC<br>VCC<br>VCC|Pretrain<br>LSTM+rep<br>LSTM(Large)|0.505<br>0.501<br>0.355<br>**0.538**<br>**0.533**<br>**0.378**<br>0.484<br>0.482<br>0.338|0.863<br>**0.829**<br>**0.643**<br>**0.863**<br>0.820<br>0.625<br>0.842<br>0.811<br>0.616|
|ES-TTS<br>ES-TTS<br>ES-TTS|Pretrain<br>LSTM+rep<br>LSTM(Large)|0.404<br>0.288<br>0.201<br>0.296<br>0.243<br>0.169<br>**0.419**<br>**0.320**<br>**0.224**|0.632<br>0.44<br>0.333<br>0.576<br>0.358<br>0.289<br>**0.655**<br>**0.527**<br>**0.422**|
|BLZ<br>BLZ<br>BLZ|Pretrain<br>LSTM+rep<br>LSTM (Large)|0.268<br>0.321<br>0.219<br>**0.323**<br>**0.328**<br>**0.226**<br>0.316<br>0.314<br>0.215|**0.487**<br>**0.557**<br>**0.384**<br>0.481<br>0.506<br>0.360<br>0.480<br>0.529<br>0.367|



with the voice conversion 2018 challenge and MOSNet (ft) is finetuned on the VoiceMOS 2022 challenge dataset. DNSMOS is trained
on the noise-suppression domain with ground truth human evaluation scores. DNSMOS predicts three scores, signal quality (SIG),
background quality (BAK), and overall quality (OVRL). We report
the correlation with human evaluations and compare against supervised baselines using three metrics: Linear Correlation Coefficient
(LCC) [38], Spearman Rank Correlation Coefficient (SRCC) [39]
and Kendall Tau (KTAU) rank correlation.


**4.2. Experimental results**


_4.2.1. Analyzing best configuration on pretrained uLM_


Different layers of self-supervised features can be important for different tasks [21, 40]. Another design choice for uLM is the number of tokens. Hence, we first compare different configurations of
SpeechLMScore using pretrained GSLM. We mainly vary the layer
number ( _L_ ) of the Hubert model to extract features and the number
of units ( _V_ ). The results are reported in Table 1. For this test, we
use the whole dataset of the VoiceMOS 2022 challenge. We also
varied temperature ( _temp_ ) for computing probability but saw little to no effect and hence use _temp_ = 1 throughout. We observe
lower layers were better correlated with human evaluations: for example _L_ = 3 and 4 were consistently better than _L_ = 12. Also
50 ~~3~~ _,_ 50 ~~4~~ _,_ and 200 ~~3~~ have highest correlation in different metrics.
For the rest of the paper, we use 50 ~~3~~ as the default SpeechLMScore
tokenizer model, unless otherwise mentioned.


_4.2.2. Correlation with Voice Conversion and TTS_


We report correlation with human evaluation on the VoiceMOS 2022
challenge testset of 1066 files and for all files 7106 files in Table 2.
We compare with two baselines: MOSNet, which is trained on the
task and dataset and DNSMOS, trained on a separate task. Since
MOSNet (ft) is trained using 4974 utterances (training set) from
the VoiceMOS dataset, we only include MOSNet (ft) results on the
testset. MOSNet finetuned on the challenge dataset has the highest
correlation to the testset. With DNSMOS scores, the signal quality
score has the highest correlation with naturalness MOS. SpeechLMScores show higher correlation than all DNSMOS scores but lower
than MOSNet. Furthermore, SpeechLMScore with duration information of tokens improve correlation. SpeechLMScore shows encouraging correlations regardless of not using target scores at all.
Next, we analyze the correlation on the three distinct sources of
the VoiceMOS dataset: voice conversion challenges (VCC), ESPnettts (ES-TTS), and Blizzard challenges (BLZ) as shown in Table 3.



**Table 4** . Utterance-level correlation with signal-to-noise ratio in
DNS-challenge-2020 noisy reverb testset [17]


Model LCC SRCC KTAU


Matched training domain


DNSMOS 0.349 0.352 0.248


Mismatched training domain


MOSNet (Pre) 0.286 0.324 0.221
MOSNet (ft) 0.443 0.456 0.325


Unsupervised


SpeechLMScore (Pre) 0.448 0.437 0.310
SpeechLMScore (Pre) (50 ~~4~~ ) 0.475 0.496 0.339
SpeechLMScore (LSTM) 0.477 0.461 0.332
SpeechLMScore (LSTM) (50 ~~4~~ ) **0.538** **0.518** **0.374**
SpeechLMScore (LSTM) + rep 0.452 0.433 0.308
SpeechLMScore (LSTM) + rep (50 ~~4~~ ) 0.478 0.455 0.324
SpeechLMScore (LSTM) (Large) 0.485 0.472 0.340
SpeechLMScore (LSTM) (Large) (50 ~~4~~ ) 0.527 0.506 0.364


We observe SpeechLMScore is highly correlated with human evaluation in voice challenge datasets, followed by both TTS systems.
However, among the TTS systems, we observe, from the Blizzard
challenge samples, the lowest correlation with older TTS systems.
Upon further inspection, we observe SpeechLMScore to have bias
towards speech fluency more than pitch characteristics which is the
main factor of evaluation in older TTS models [41].


_4.2.3. Correlation with noise in speech_


Next, we evaluate if SpeechLMScore correlates with the presence of
noise in speech. For this experiment, we use Deep Noise Suppression (DNS) challenege-2020 [17] noisy synthetic testset with reverberation. We measure the correlation with signal-to-noise ratio of
the speech samples and report in Table 4 with both supervised baseline MOSNet and DNSMOS. We observe SpeechLMScore achieves
a higher correlation with signal-to-noise ratio than both DNSMOS
and MOSNet. Therefore, we conclude that such automatic unsupervised metric can generalize to different speech evaluation tasks.


**5. CONCLUSIONS**


We propose SpeechLMScore, an automatic metric for evaluating
speech samples using speech language models. Proposed metric is
simple, easy to use and does not require reference speech sample.
Furthermore, SpeechLMScore is trained using speech dataset only,
and does not need large-scale human evaluation data. We show our
proposed metric is highly correlated with human evaluation in different speech systems like voice conversion, text-to-speech using the
VoiceMOS challenge dataset. We also show SpeechLMScore correlates with SNR using DNS 2020 challenge testset. Though proposed
metric is generalizable to different tasks, there is scope for further
improvement. Future extensions are envisioned to use more complex and prosody-rich tokens.


**Acknowledgements:** This work used the Extreme Science and
Engineering Discovery Environment (XSEDE) [42], which is supported by National Science Foundation grant number ACI-1548562.
Specifically, it used the Bridges system [43], which is supported by
NSF award number ACI-1445606, at the Pittsburgh Supercomputing
Center (PSC).


**6. REFERENCES**


[1] T. Fukada, K. Tokuda, T. Kobayashi, et al., “An adaptive algorithm
for mel-cepstral analysis of speech.,” in _Proc. ICASSP_ . Citeseer, 1992,
vol. 92, pp. 137–140.


[2] A. W Rix, J. G Beerends, M. P Hollier, et al., “Perceptual evaluation
of speech quality (pesq)-a new method for speech quality assessment
of telephone networks and codecs,” in _Porc. ICASSP_, 2001, vol. 2.


[3] C.-C. Lo, S.-W. Fu, W.-C. Huang, et al., “MOSNet: Deep learning
based objective assessment for voice conversion,” in _Proc. Interspeech_,
2019, pp. 1541–1545.


[4] T. Sellam, A. Bapna, J. Camp, et al., “SQuId: Measuring speech naturalness in many languages,” _arXiv preprint arXiv:2210.06324_, 2022.


[5] C. K A Reddy, V. Gopal, and R. Cutler, “Dnsmos: A non-intrusive perceptual objective speech quality metric to evaluate noise suppressors,”
in _Proc. ICASSP_, 2021, pp. 6493–6497.


[6] E. Cooper, W.-C. Huang, T. Toda, et al., “Generalization ability of mos
prediction networks,” in _Proc. ICASSP_, 2022, pp. 8442–8446.


[7] W.-C. Huang, E. Cooper, Y. Tsao, et al., “The voicemos challenge
2022,” _arXiv preprint arXiv:2203.11389_, 2022.


[8] K. Lakhotia, E. Kharitonov, W.-N. Hsu, et al., “On generative spoken
language modeling from raw audio,” _Transactions of the Association_
_for Computational Linguistics_, vol. 9, pp. 1336–1354, 2021.


[9] E. Kharitonov, A. Lee, A. Polyak, et al., “Text-free prosody-aware generative spoken language modeling,” _arXiv preprint arXiv:2109.03264_,
2021.


[10] Z. Borsos, R. Marinier, D. Vincent, et al., “Audiolm: a language modeling approach to audio generation,” _arXiv preprint arXiv:2209.03143_,
2022.


[11] W. Zhao, M. Peyrard, F. Liu, et al., “MoverScore: Text generation
evaluating with contextualized embeddings and earth mover distance,”
in _Proc. EMNLP-IJCNLP_, 2019, pp. 563–578.


[12] T. Zhang, V. Kishore, F. Wu, et al., “Bertscore: Evaluating text generation with bert,” _arXiv preprint arXiv:1904.09675_, 2019.


[13] W. Yuan, G. Neubig, and P. Liu, “Bartscore: Evaluating generated text
as text generation,” _Proc NeurIPS_, vol. 34, pp. 27263–27277, 2021.


[14] K. Papineni, S. Roukos, T. Ward, et al., “Bleu: a method for automatic
evaluation of machine translation,” in _Proc. ACL_, 2002, pp. 311–318.


[15] T. Hayashi and S. Watanabe, “Discretalk: Text-to-speech as a machine
translation problem,” _arXiv preprint arXiv:2005.05525_, 2020.


[16] B. Patton, Y. Agiomyrgiannakis, M. Terry, et al., “AutoMOS: Learning a non-intrusive assessor of naturalness-of-speech,” _arXiv preprint_
_arXiv:1611.09207_, 2016.


[17] C. KA Reddy, V. Gopal, R. Cutler, et al., “The interspeech 2020 deep
noise suppression challenge: Datasets, subjective testing framework,
and challenge results,” in _Proc. Interspeech_, 2020.


[18] T. Saeki, D. Xin, W. Nakata, et al., “UTMOS: UTokyo-SaruLab System
for VoiceMOS Challenge 2022,” in _Proc. Interspeech_, 2022, pp. 4521–
4525.


[19] P. Manocha and A. Kumar, “Speech Quality Assessment through MOS
using Non-Matching References,” in _Proc. Interspeech_, 2022, pp. 654–
658.


[20] M. Rivi`ere and E. Dupoux, “Towards unsupervised learning of speech
features in the wild,” in _Proc. SLT_, 2021, pp. 156–163.


[21] W.-N. Hsu, B. Bolte, Y.-H. Tsai, et al., “HuBERT: Self-supervised
speech representation learning by masked prediction of hidden units,”
_IEEE/ACM Transactions on Audio Speech and Language Processing_,
2021.


[22] A. Baevski, Y. Zhou, A. Mohamed, et al., “wav2vec 2.0: A framework for self-supervised learning of speech representations,” in _Proc._
_NeurIPS_, 2020, vol. 33, pp. 12449–12460.


[23] J. A Hartigan and M. A Wong, “Algorithm as 136: A k-means clustering algorithm,” _Journal of the Royal Statistical Society. Series C_
_(Applied Statistics)_, vol. 28, pp. 100–108, 1979.




[24] A. Vaswani, N. Shazeer, N. Parmar, et al., “Attention is all you need,”
in _Advances in Neural Information Processing Systems_, 2017.


[25] S. Hochreiter and J. Schmidhuber, “Long short-term memory,” _Neural_
_Computation_, vol. 9, no. 8, pp. 1735–1780, 1997.


[26] T. Toda, L.-H. Chen, D. Saito, et al., “The voice conversion challenge
2016.,” in _Proc. Interspeech_, 2016, pp. 1632–1636.


[27] J. Lorenzo-Trueba, J. Yamagishi, T. Toda, et al., “The voice conversion challenge 2018: Promoting development of parallel and nonparallel methods,” 2018.


[28] Z. Yi, W.-C. Huang, X. Tian, et al., “Voice conversion challenge 2020intra-lingual semi-parallel and cross-lingual voice conversion,” in _Proc._
_Joint Workshop for the Blizzard Challenge and Voice Conversion Chal-_
_lenge 2020_, 2020, pp. 80–98.


[29] T. Hayashi, R. Yamamoto, K. Inoue, et al., “ESPnet-TTS: Unified,
reproducible, and integratable open source end-to-end text-to-speech
toolkit,” in _Proc. ICASSP_, 2020, pp. 7654–7658.


[30] Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur, “Librispeech: An asr corpus based on public domain audio books,”
in _Proc. ICASSP)_, 2015, pp. 5206–5210.


[31] S. King, R. AJ Clark, C. Mayo, et al., “The blizzard challenge 2008,”
2008.


[32] S. King and V. Karaiskosb, “The blizzard challenge 2009,” 2009.


[33] K. Prahallad, A. Vadapalli, N. Elluru, et al., “The blizzard challenge
2013–indian language task,” 2013.


[34] Gregor Pirker, Michael Wohlmayr, Stefan Petrik, and Franz Pernkopf,
“A pitch tracking corpus with evaluation on multipitch tracking scenario,” in _Proc. Interspeech_, 2011.


[35] Jort F. Gemmeke, Daniel P. W. Ellis, Dylan Freedman, Aren Jansen,
Wade Lawrence, R. Channing Moore, Manoj Plakal, and Marvin Ritter,
“Audio set: An ontology and human-labeled dataset for audio events,”
in _Proc. ICASSP_, 2017, pp. 776–780.


[36] J. Kahn, M. Rivi`ere, W. Zheng, et al., “Libri-light: A benchmark for
asr with limited or no supervision,” in _Proc. ICASSP_, 2020, pp. 7669–
7673.


[37] Diederik P Kingma and Jimmy Ba, “Adam: A method for stochastic
optimization,” _arXiv preprint arXiv:1412.6980_, 2014.


[38] K. Pearson, “Notes on the history of correlation,” _Biometrika_, vol. 13,
no. 1, pp. 25–45, 1920.


[39] C. Spearman, “The proof and measurement of association between two
things,” _The American Journal of Psychology_, vol. 100, no. 3/4, pp.
441–471, 1987.


[40] Shu-wen et al. Yang, “Superb: Speech processing universal performance benchmark,” in _Proc. Interspeech_, 2021.


[41] S. L. Maguer, S. King, and N. Harte, “Back to the Future: Extending
the Blizzard Challenge 2013,” in _Proc. Interspeech_, 2022, pp. 2378–
2382.


[42] J. Towns, T. Cockerill, M. Dahan, et al., “XSEDE: Accelerating scientific discovery,” _Computing in Science & Engineering_, vol. 16, no. 5,
pp. 62–74, 2014.


[43] Nicholas A Nystrom, Michael J Levine, Ralph Z Roskies, and J Ray
Scott, “Bridges: a uniquely flexible hpc resource for new communities
and data analytics,” in _Proc. 2015 XSEDE Conference: Scientific Ad-_
_vancements Enabled by Enhanced Cyberinfrastructure_, 2015, pp. 1–8.


