## **Automatic Proficiency Assessment in L2 English Learners**

Armita Mohammadi [1], Alessandro Lameiras Koerich [1], Laureano Moro-Velazquez [2] and Patrick Cardinal [1]



_**Abstract**_ **— Second language proficiency (L2) in English is**
**usually perceptually evaluated by English teachers or expert**
**evaluators, with the inherent intra- and inter-rater variability.**
**This paper explores deep learning techniques for comprehen-**
**sive L2 proficiency assessment, addressing both the speech**
**signal and its correspondent transcription. We analyze spoken**
**proficiency classification prediction using diverse architectures,**
**including 2D CNN, frequency-based CNN, ResNet, and a pre-**
**trained wav2vec 2.0 model. Additionally, we examine text-based**
**proficiency assessment by fine-tuning a BERT language model**
**within resource constraints. Finally, we tackle the complex**
**task of spontaneous dialogue assessment, managing long-form**
**audio and speaker interactions through separate applications**
**of wav2vec 2.0 and BERT models. Results from experiments**
**on EFCamDat and ANGLISH datasets and a private dataset**
**highlight the potential of deep learning, especially the pre-**
**trained wav2vec 2.0 model, for robust automated L2 proficiency**
**evaluation.**


I. INTRODUCTION


The accurate assessment of second language (L2) proficiency is crucial in education. Traditionally, this has relied
on standardized tests and human evaluation, exemplified
by the test of English as a foreign language (TOEFL),
the international English language testing system (IELTS),
and Cambridge English assessments, which are often costly,
time-consuming, and subjective [1]. To overcome these limitations, automated assessment technologies have emerged,
aiming for scalable and objective evaluations. Early efforts
focused on written language, utilizing rule-based and statistical models [2]. However, spoken proficiency, particularly in
spontaneous dialogue, poses a significantly greater challenge
due to its inherent variability [3].
Initial automated spoken assessment systems employed
feature-based models, extracting handcrafted linguistic and
acoustic features via automatic speech recognition (ASR) to
quantify proficiency aspects like fluency and pronunciation

[3], [4]. While systematic, these models struggled with contextual meaning and interactional competence, highlighting
the need for methods that capture the dynamic nature of realworld conversations.

Recent advancements in machine learning (ML) and
deep learning (DL) have transformed automated language
assessment. Neural networks and pretrained models like
wav2vec 2.0 [5] have shown promising performance in
analyzing monologic speech, as seen in tests like TOEFL


*This work was supported by a MITACS Accelerate grant.
1A. Mohammadi, A. Lameiras Koerich and P. Cardinal are
with ´Ecole de Technologie Sup´erieure, Universit´e du Qu´ebec,
Montreal, QC, Canada. armita.mohammadi.1@ens.etsmtl.ca,

alessandro.koerich@etsmtl.ca, patrick.cardinal@etsmtl.ca
2L. Moro-Velazquez is with Johns Hopkins University, Baltimore, MD,
USA. laureano@jhu.edu



iBT, Linguaskill Speaking, and the Duolingo English test.
However, dialogic speech assessment remains a significant
hurdle. Evaluating interactional competence and real-time
responsiveness requires models beyond traditional metrics

[6].
Current systems primarily focus on structured speech
tasks, limiting their applicability to unscripted communication. Deep learning models, while effective in structured
speech classification, require further investigation in spontaneous settings. Similarly, adapting text-based models like
BERT [7] for spoken language presents unique challenges
due to differences in discourse and tokenization.

This research addresses these limitations through a progressive approach, from structured to dialogic speech, by:
(i) evaluating CNNs [8], ResNets [9], and wav2vec 2.0
for L2 proficiency classification on structured speech from
the ANGLISH dataset, (ii) assessing BERT’s performance
on transcribed speech in a low-resource setting, and (iii)
developing methods for evaluating spontaneous dialogue
proficiency. In summary, this study provides a comprehensive
and robust approach to automated L2 assessment by analyzing fluency and communicative competence in unstructured
conversations.

**Our main contributions are as follows:**


_•_ We evaluate a diverse set of deep learning models
for L2 proficiency classification on both structured and
spontaneous speech, spanning multiple datasets.

_•_ We propose a dialogue-specific preprocessing pipeline
to effectively process long-form, multi-speaker audio.

_•_ We conduct all experiments under a strict speakerindependent setup to ensure a realistic assessment of
model generalizability.

_•_ We investigate multi-task learning for the joint prediction of proficiency level and speaker gender, demonstrating improvements over single-task baselines.

_•_ We examine how input segmentation strategies influence
the performance of text-based classification models.


II. RELATED WORK


The automated assessment of spoken English proficiency
has evolved significantly, driven by the need for scalable and objective evaluation methods. Early approaches,
rooted in contrastive linguistics and error analysis [10],

[11], transitioned toward communicative competence frameworks, emphasizing meaningful communication [12]–[14].
The Common European Framework of Reference for Languages (CEFR) further formalized this shift, integrating error
analysis with communicative competence [15], [16].


Initial automated spoken language assessment (ASLA)
systems relied on handcrafted features like MFCCs and
fluency metrics, trained with SVMs [17] and RNNs [18]–

[20]. These systems demonstrated the feasibility of automating scoring, achieving consistency comparable to human
raters. For instance, models proposed by Zechner and Bejar

[18] achieved 47.4% accuracy for five proficiency classes
on integrated tasks, surpassing human agreement rates of
43%. However, they faced limitations due to manual feature
engineering and ASR challenges.
Deep learning advancements enabled end-to-end learning
from raw data. Models like BD-LSTMs and BD-RNNs

improved performance, integrating acoustic and lexical features [21], [22]. Recent work highlights the efficacy of selfsupervised learning (SSL) models, such as wav2vec 2.0,
which outperform traditional methods in spontaneous speech
tasks by leveraging raw audio [23]–[26]. Hybrid models,
combining SSL, BERT-based text systems, and handcrafted
features, have shown the best results, balancing efficiency
and nuanced feedback [6].
In text-based assessment, BERT-based frameworks have
been explored to enrich contextual embeddings, achieving
high accuracy on large datasets [27], [28]. Multimodal approaches, like combining wav2vec 2.0 and Longformer, have
further improved dialogic assessment [6]. Hierarchical graphbased models and multi-task learning frameworks, which
incorporate sentiment analysis and coherence modeling, have
also shown promising results [29]–[31]. Nebhi and Szasz´ak

[14] demonstrated a multimodal multitask transformer model
for CEFR-aligned assessment, achieving high accuracy.
While several studies have explored wav2vec 2.0 for L2
assessment, many rely on speaker-dependent setups. Our
study specifically emphasizes speaker-independent evaluation, a more realistic and challenging scenario for deploymen


III. PROPOSED APPROACH


_A. Preprocessing and Segmentation_


Models like 2D CNNs and ResNet require the preprocessing of the speech signal. The preprocessing pipeline
consisted of multiple stages, including audio downsampling
to 16 kHz, segmentation into 8-sec excerpts for models
requiring fixed length inputs, feature extraction using FBanks
with 40 Mel-frequency bins, and input normalization using

z-score.


_B. Dialogue-Specific Preprocessing_


To evaluate the impact of interviewer speech on classification performance, we applied a preprocessing pipeline to separate learner and interviewer segments. Silence was removed
using Pydub [32], with parameters set to a minimum silence
length of 500 ms, a threshold of _−_ 40 dB, and a 100 ms
buffer to preserve natural speech flow. Speaker diarization
followed a two-step approach: initial segmentation using
PyAnnote [33] and refinement with SpeechBrain [34] and
following McKnight et al. [6], the learner was identified as
the speaker with the longest total speaking time. This allowed
us to create two dataset versions: one with full dialogues and



another containing only learner speech. Since the recordings
lacked transcripts, we used Whisper [35] to generate automatic speech transcriptions and evaluated multiple variants
to identify the most accurate for our data. Despite its strong
performance, Whisper occasionally produced errors due to
overlapping speech and real-world noise conditions. A subset
of the data was manually reviewed to verify segmentation

accuracy.


_C. Models_


**2D CNN** : The architecture consists of four convolutional layers with ReLU activation function to introduce non-linearity,
thus enhancing the model’s ability to detect robust features.
To prevent overfitting and improve model generalization,
batch normalization, max pooling, and dropout layers are
applied after each convolutional operation.
**Frequency-Axis CNN** : Using the same architecture, we
apply convolutions specifically along the frequency axis to
capture fine-grained spectral features, such as harmonics and
fundamental frequency. This enhances the model’s ability to
detect key audio characteristics while maintaining regularization through batch normalization, max pooling, and dropout.
**ResNet** : We employed a 2D CNN architecture with five
residual blocks, featuring channel configurations of 128, 128,
256, 256, and 512. Each block uses 2D convolutional layers
to process the spectrogram generated by the FBanks. The
convolutional layers have a kernel size of 3 and a stride of
2 to downsample the input. The residual connection links
the output from the downsampling step to the output of
the last convolutional layer within each block. After the
convolutional operations, global average pooling is applied
before passing the results to a fully connected layer for
classification.
**wav2vec 2.0** : The models were initialized with a pre-trained
model [1] . Learners’ audio responses were input to the model,
where the transformer encoder generated contextualized
representations. We fine-tuned the wav2vec 2.0 model for
our classification tasks. Finally, the encoder’s output was
aggregated using a _StatisticsPooling_ layer with mean pooling,
and two linear classifiers produced the final predictions. To
enhance generalization and mitigate overfitting, we froze
the convolutional feature extractor layer of the wav2vec
2.0 model and fine-tuned its transformer encoder. Separate
optimizers were employed for the transformer encoder and
the linear classifier, ensuring optimal learning rates for each
component. This fine-tuning strategy enables the model to
adapt its pretrained representations to the specific requirements of the classification tasks.
**BERT** : We explored text-based classification using
transcriptions of learner responses. A pre-trained
bert-base-uncased model was used to generate
contextual embeddings, with its parameters frozen during
training to prevent overfitting.


1https://huggingface.co/patrickvonplaten/wav2vec2-base


IV. EXPERIMENTAL RESULTS

_A. Datasets_

**ANGLISH:** This corpus [36] was designed to analyze British
English as a second language (L2) among native French
speakers. It comprises over 5.5 hours of speech (20,396
seconds) from a variety of structured and spontaneous speech
tasks by 67 participants across three proficiency levels: (i)
native English speakers (NES), the highest proficiency group,
including 13 females and 10 males with an average age of 31
years from various regions in England; (ii) French speakers
subdivided into 11 females and 11 males with an average
age of 37.5 years without specialized phonetics training
(FR1) and university students, 11 females and 11 males, aged
between 19 to 22 years with formal phonetics training (FR2).
All recordings were conducted in an anechoic chamber to
ensure pristine audio quality. We used the reading passages,
where participants read four texts, yielding approximately
1.5 hours of speech across 1,260 sentences, and spontaneous
monologues, where 63 participants delivered 4-minute talks
on pre-suggested topics (e.g., holidays). The audio data
was transcribed and segmented at the phoneme level using
PRAAT [37], allowing the capture of nuanced linguistic
features such as pauses, hesitations, and truncated words.
The corpus is publicly available for academic use [38].
**EFCamDat:** The EF-Cambridge open language dataset comprises 1,180,310 essays written by learners and subsequently
corrected and evaluated by professional language instructors

[39]. It was created from a global pool of English language
learners spanning over 172 nationalities. The dataset represents 16 different proficiency levels mapped to five common
European framework of reference for languages (CEFR)
levels: A1, A2, B1, B2, and C1, which reflect the progression
from beginner (A1) to advanced (C1) proficiency, making
the EFCamDat a highly diverse and stratified resource for
studying learner language. Each essay is accompanied by
rich metadata, including the learner’s native language, age,
gender, and time spent learning English.
**Private:** The private dataset contains structured interview
recordings to assess students’ language proficiency. These
recordings are dialogue-based, where students respond to
interviewer-led questions and proficiency is evaluated across
four key dimensions: fluency, comprehension, accuracy,
and vocabulary, with an overall score assigned to each
recording. The dataset is divided into two sections: (i)
_tests_ _linguistiques_, contains learners’ initial proficiency levels, ranging from 1 to 8; (ii) _tests_ ~~_v_~~ _alidation_ evaluates
proficiency after language coursework. We used only part of
the _tests_ ~~_l_~~ _inguistiques_ section, levels L3 to L5, which data
distribution across proficiency levels in show in Table I. Such
levels provided the largest number of speakers and samples.


_B. Experimental Protocol_

**ANGLISH:** We employed 10-fold cross-validation (CV)
using 61 speakers. In each fold, a subset of speakers was
used for training, while a distinct subset was held out for
validation. Stratification ensured balanced gender and proficiency representation across folds. We also have a test set



TABLE I

THE DISTRIBUTION OF DATA ACROSS DIFFERENT LEVELS OF THE


PRIVATE DATASET

|Level|Duration|Col3|Col4|Numberof<br>Speakers|
|---|---|---|---|---|
|**Level**|**Total**|**Min**|**Max**|**Max**|
|L1|3:44:31|6:29|17:45|19|
|L2|27:03:08|4:11|25:15|135|
|L3|54:17:21|4:06|24:25|225|
|L4|50:35:31|5:14|24:41|197|
|L5|35:50:34|6:22|27:40|126|
|L6|19:00:50|9:08|36:14|60|
|L7|9:34:04|12:11|27:61|29|
|L8|2:06:32|19:38|33:12|5|



composed of 6 speakers (one male and one female from each
proficiency level), never seen during training or validation,
to assess the generalization of the models.
**EFCamDat:** In our experiments, we utilized a carefully
curated subset with 2,000 samples for the training set, 200
samples for the validation set, and 200 samples for the test
set. To ensure representativeness and fairness, we stratified
the subset according to the 16 proficiency levels present
in the full dataset. This stratification helps preserve the
proportional distribution of samples across levels, enabling
a robust analysis of proficiency classification.
**Private:** To ensure a fair and robust evaluation, we partitioned the data into training, validation, and test sets.
For each level, we randomly selected 12 speakers for the
validation and test sets, ensuring that their total audio duration matched 10% of the total duration in L5, the least
represented level. The remaining speakers were used for
training. Crucially, we enforced strict speaker separation
across partitions, ensuring that no speaker appeared in both
training and evaluation sets. This approach guarantees that
model performance is assessed on entirely unseen speakers,
providing a more reliable measure of generalizability.


_C. Implementation Details_


**Anglish:** _2D CNN & Frequency-Axis CNN:_ Trained for 200
epochs, batch size 8, learning rate decayed from 0.001 to
0.0001, with early stopping (patience = 10). _ResNet:_ 100
epochs, batch size 16, same learning rate schedule. _wav2vec_
_2.0:_ Fine-tuned for 5 epochs, batch size 16, using learning
rates of 1e _[−]_ [4] (classifier) and 1e _[−]_ [5] (encoder), encoder dimension 768. _Multi-task:_ A dual-layer DNN (LeakyReLU,
batch norm, log-softmax) was used for joint gender (binary)
and proficiency (3-class) classification, with loss _L_ final =
3 _L_ level + _L_ gender. All models used Adam (weight decay =
0.0001).
**EFCamDat:** _BERT+SVM:_ Fixed BERT embeddings with
an RBF-kernel SVM ( _C_ = 10, _γ_ = 0 _._ 01), trained via
SGD (hinge loss, L2 regularization = 1e _[−]_ [3] ). _BERT+MLP:_
A 3-layer MLP (768, 128, 5) with ReLU and 0.2 dropout,
trained for 300 epochs using Adam (1e _[−]_ [5] ), early stopping
(patience = 10). _Fine-tuned BERT+MLP:_ Last BERT layers
and MLP fine-tuned for 30 epochs (patience = 2). _Token_
_Length:_ We tested input lengths from 10 to 90 tokens to
evaluate performance–efficiency trade-offs.


**Private:** _2D CNN & ResNet:_ Trained for 100 epochs,
batch size 16, with learning rate decay (0.001 to 0.0001),
and early stopping. _wav2vec 2.0:_ Aggregated 2D features
(mean pooling), followed by a 768-unit dense layer, 0.2
dropout, softmax output. Trained for 20 epochs using
AdamW (batch size 4, learning rate 1e _[−]_ [5] ). _BERT:_ Frozen
bert-base-uncased with BiLSTM, self-attention, and
dense output. Two input formats: (1) 1-min transcript, (2)
7-sentence segments. Trained for 120 epochs (learning rate:
0.001, weight decay: 0.01).


_D. Results_


**ANGLISH:** Performance was evaluated separately for
single-task learning (proficiency or gender classification
alone) and multi-task learning (joint proficiency and gender classification). The results for each model under these
settings are presented in Table II.


TABLE II

RESULTS OF DIFFERENT MODELS ON THE TEST SET OF THE ANGLISH


DATASET.

|Model|LevelACC<br>Single-task|LevelACC<br>Multi -task|
|---|---|---|
|2D CNN|20.8%|29.2%|
|Frequency-Axis 2D CNN|27.4%|33.3%|
|ResNet|35.9%|43.8%|
|wav2vec 2.0|75%|80.0%|



The experimental results reveal significant performance
differences across the evaluated models, with pretrained
wav2vec 2.0 emerging as the top-performing architecture.
Notably, wav2vec 2.0 achieved 75% accuracy in the singletask proficiency classification task and 80% in the multitask setting, demonstrating its superior ability to generalize
and capture intricate speech patterns. This performance advantage can be attributed to its pretrained representations,
which are adept at encoding complex acoustic features and
leveraging large-scale self-supervised learning.
**EFCamDat:** Table III presents the average lengths of essays
across different CEFR proficiency levels, highlighting the
variation in text complexity and length as learners progress
from A1 to C1. Table III illustrates that lower-level learners


TABLE III

LEVEL DISTRIBUTION OF THE EFCAMDAT DATASET WITH AVERAGE


SEQUENCE LENGTH

|Level|#ofAnswers|AverageLength|
|---|---|---|
|A1|191,663|40|
|A2|129,591|67|
|B1|61,506|92|
|B2|18,187|129|
|C1|5,115|179|



(A1, A2) tend to write shorter essays with less variability in
length, reflecting their limited vocabulary and writing skills.
Conversely, higher-level learners (B2, C1) produce longer
and more complex essays with greater variability, indicative
of their more advanced language proficiency and diverse
use of linguistic structures. This variability underscores the



need for adaptive preprocessing strategies, particularly when
working with texts of varying lengths. The performance of
different classifiers using a fixed input length of 60 tokens
is summarized in Table IV. These results offer insights into
the effectiveness of various classification methods applied to
BERT embeddings.


TABLE IV


PERFORMANCE OF THE BERT MODELS WITH DIFFERENT CLASSIFIERS


AND FINE-TUNED BERT MODEL WITH MAX SEQUENCE LENGTH OF 60

|ModelName|Accuracy|Precision|F1|
|---|---|---|---|
|BERT+SVM|79.0%|0.798|0.791|
|BERT+MLP|86.5%|0.88|0.89|
|FTBERT+MLP|87.5%|0.89|0.90|



Table IV shows a clear trend of improvement in performance as we move from the simpler BERT+SVM model
to the more advanced FTBERT+MLP model. The accu
racy, precision, and F1 score all increase with each model,
demonstrating the impact of incorporating more sophisticated classification techniques. The comparative analysis of the models reveals distinct performance trends for
BERT+SVM, BERT+MLP, and FTBERT+MLP across varying token lengths. The results, summarized in Table V,
highlighting the impact of token length on model accuracy
and the benefits of fine-tuning.
**Private:** Model performance was evaluated using classification accuracy, macro precision, macro recall, and macro F1score. Each model was trained on a training set and evaluated
on an independent test set. The results, presented in Table VI,
compare the performance of audio-based models, including
CNN, ResNet, and wav2vec 2.0 variants, against text-based
models using BERT and highlight the impact of model
architecture, input duration, and segmentation strategies on
classification performance.
The results demonstrate the superiority of wav2vec 2.0
over traditional CNN-based architectures for proficiency
classification. Both CNN and ResNet models performed
poorly, with accuracies of 29.2% and 31.4%, respectively,
underscoring their limitations in modeling the temporal
complexity of spontaneous speech. In contrast, wav2vec 2.0
models achieved significantly higher performance, with the
8-second full-audio model reaching 60.5% accuracy and the
60-second variant peaking at 62.0%. These results suggest
that longer audio segments contribute richer contextual cues,
although performance gains begin to saturate beyond 30
seconds.

A key insight from the comparison of full-audio (FA)
and student-only audio (SA) models is that incorporating
the interviewer’s speech improves classification. SA models
consistently underperformed, with the 60-second studentonly model achieving 61.0%, compared to 62.0% for the fullaudio counterpart. This highlights the contribution of interactional dynamics—such as turn-taking, interviewer prompts,
and response timing—in enhancing audio-based classification.

In contrast, for text-based models, using only the student’s


TABLE V

ACCURACY (%) OF BERT MODELS AT VARIOUS TOKEN LENGTHS


|Models|NumberofTokens|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|
|---|---|---|---|---|---|---|---|---|---|
|**Models**|**10**|**20**|**30**|**40**|**50**|**60**|**70**|**80**|**90**|
|BERT+SVM|68.5|71.5|79.5|81.0|82.5|83.5|84.0|85.5|86.0|
|BERT+MLP|66.0|73.5|86.5|84.0|84.0|85.0|84.0|86.5|87.0|
|FTBERT+MLP|70.0|83.0|87.5|88.5|92.0|88.5|88.5|91.5|92.0|



TABLE VI

CLASSIFICATION ACCURACY OF AUDIO-BASED AND TEXT-BASED


MODELS ON THE TEST SET OF THE PRIVATE DATASET

|Model|Accuracy|Macro<br>Precision|Macro<br>Recall|Macro<br>F1|
|---|---|---|---|---|
|CNN|29.2%|NC|NC|NC|
|ResNet|31.4%|NC|NC|NC|
|wav2vec 2.0 FA 8-sec|60.5%|0.58|0.57|0.56|
|wav2vec 2.0 FA 30-sec|60.0%|0.59|0.59|0.58|
|wav2vec 2.0 FA 60-sec|62.0%|0.61|0.63|0.61|
|wav2vec 2.0 SA 30-sec|55.0%|0.63|0.58|0.54|
|wav2vec 2.0 SA 60-sec|61.0%|0.62|0.61|0.60|
|BERT FA 1-min|45%|0.49|0.45|0.43|
|BERT FA 7-sent|52%|0.54|0.51|0.52|
|BERT SA 1-min|56%|0.54|0.52|0.52|
|BERT SA 7-sent|57%|0.58|0.55|0.56|



FA: Full audio. SA: Student audio. NC: Not computed


transcript yielded better results. The best-performing text
model was BERT SA 7-sentences, with an accuracy of
57% and a Macro F1 of 0.56, outperforming its full-audio
counterpart (BERT FA 7-sentences), which reached only
52%. This indicates that including the interviewer’s text may
introduce noise or irrelevant content, while shorter, studentonly segments provide more focused and discriminative
information. Notably, reducing input length from one-minute
transcripts to 7-sentence excerpts improved results across
both FA and SA models.

Overall, audio-based wav2vec 2.0 models outperformed
their BERT-based textual counterparts, with the top audio
model achieving 62% accuracy versus 57% for text. These
findings underscore the importance of leveraging speechspecific features for proficiency assessment, while also highlighting modality-specific best practices: include full context
in audio but focus on concise, student-only content in text.


V. DISCUSSION


Overall, our findings highlight the superiority of wav2vec
2.0 for speech-based proficiency assessment, demonstrating that while text-based models provide useful linguistic
insights, they are less effective in isolation. The higher
misclassification rates of BERT models, particularly in distinguishing adjacent proficiency levels, suggest that text
alone lacks crucial prosodic and fluency-related cues. Future research could explore multimodal fusion strategies,
integrating speech and text-based features to capitalize on
their complementary strengths. Such an approach has the
potential to enhance classification accuracy, particularly for
proficiency levels where lexical, syntactic, and prosodic
features jointly contribute to speaker differentiation.



Compared to prior work, our results align with studies
such as Zhou et al. [40], which emphasized the importance of
diverse feature representations for proficiency scoring. While
their approach relied on manually extracted linguistic features and a Random Forest classifier, our study extends this
by leveraging deep learning models that automatically learn
high-dimensional representations from speech. Additionally,
while Lo et al. [26] reported high proficiency classification
accuracy using wav2vec 2.0, their results were achieved in
a speaker-dependent setting, whereas our study focuses on
speaker-independent generalization, a more realistic scenario
for proficiency assessment. This distinction is critical as it
highlights the limitations of certain state-of-the-art models
that may perform well in controlled conditions but struggle
with generalization.
Compared to McKnight et al. [6], which also investigated dialogue-based assessment, our study differs in approach: their work applied regression-based scoring using
a wav2vec 2.0 + Longformer model, whereas our study
primarily focused on classification-based evaluation. Their
results suggest that transformer-based architectures improve
scoring accuracy when combined with speech embeddings,
yet our findings indicate that wav2vec 2.0 alone struggles
when applied to highly spontaneous dialogues, particularly
in scenarios with greater fluency variability and disfluencies.
This suggests that while transformer-based models are beneficial for structured dialogue-based tasks, more advanced
multimodal approaches may be needed to robustly assess
conversational speech in real-world settings.


VI. CONCLUSION


This research explored deep learning methodologies for
automated L2 proficiency assessment across speech and text
modalities, addressing both structured and spontaneous language contexts. Leveraging novel datasets and state-of-theart models, we demonstrated the potential of these techniques
for robust and scalable evaluation.

For speech-based assessment, pretrained models like
wav2vec 2.0 excelled in capturing intricate speech patterns
and generalizing to unseen speakers. Multi-task learning
enhanced proficiency classification, though it highlighted the
need for specialized architectures to balance task complexities. While our speaker-independent approach addressed the
limitations of prior work, future research should integrate
multimodal features to further improve performance.
In text-based assessment, fine-tuned BERT models significantly improved classification accuracy, confirming the effectiveness of deep contextual representations. We demonstrated
strong performance even with limited data and sequence


lengths, showcasing feasibility in resource-constrained settings. Future work should explore lightweight transformer
variants and optimize sequence length effects for scalable
solutions.

Assessing spontaneous dialogues presented unique challenges due to fluency variability and disfluencies. While
wav2vec 2.0 and BERT performed well individually, fusion strategies are needed to leverage their complementary strengths. Compared to regression-based approaches,
our classification-based evaluation highlighted the need for
advanced multimodal models in real-world conversational

settings. Future research should investigate multimodal learning and hierarchical modeling techniques for robust dialogue

assessment.

Overall, this study underscores the efficacy of deep learning for automated L2 proficiency assessment. Future directions include developing specialized multi-task learning
frameworks, optimizing transformer models for resourcelimited scenarios, and exploring multimodal approaches for
spontaneous dialogue evaluation. These advancements will
pave the way for more accurate, efficient, and scalable
language assessment systems.


REFERENCES


[1] C. A. Chapelle and D. Douglas, _Assessing Language through Com-_
_puter Technology_ . Cambridge Language Assessment, Cambridge Univ
Press, 2006.

[2] J. C. Alderson, _Assessing Reading_ . Cambridge Language Assessment,
Cambridge Univ Press, 2000.

[3] K. Zechner and K. Evanini, eds., _Automated Speaking Assessment: Us-_
_ing Language Technologies to Score Spontaneous Speech_ . Routledge,
1st ed., 2019.

[4] L. Chen, K. Zechner, S.-Y. Yoon, K. Evanini, X. Wang, A. Loukina,
J. Tao, L. Davis, C. M. Lee, M. Ma, R. Mundkowsky, C. Lu, C. W.
Leong, and B. Gyawali, “Automated scoring of nonnative speech
using the speechratersm v. 5.0 engine,” _ETS Research Report Series_,
vol. 2018, 03 2018.

[5] A. Baevski, H. Zhou, A. Mohamed, and M. Auli, “wav2vec 2.0:
A framework for self-supervised learning of speech representations,”
2020.

[6] S. W. McKnight, A. Civelekoglu, M. J. Gales, S. Banno, A. Liusie,
and K. M. Knill, “Automatic assessment of conversational speaking
tests,” 2023.

[7] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training
of deep bidirectional transformers for language understanding,” 2019.

[8] K. O’Shea and R. Nash, “An introduction to convolutional neural
networks,” 2015.

[9] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for
image recognition,” 2015.

[10] R. Lado, _Language Testing: The Construction and Use of Foreign_
_Language Tests. Teacher’s Book_ . McGraw-Hill Book Comp, 1961.

[11] S. P. Corder, “The significance of learner’s errors,” _Intl Review of_
_Applied Linguistics in Language Teaching_, vol. 5, no. 1-4, pp. 161–
170, 1967.

[12] S. Whyte, “Revisiting communicative competence in the teaching and
assessment of language for specific purposes,” _Language Education &_
_Assessment_, vol. 2, pp. 1–19, 04 2019.

[13] M. Canale and M. Swain, “Theoretical bases of communicative
approaches to second language teaching and testing,” _Applied Lin-_
_guistics_, vol. I, 03 1980.

[14] K. Nebhi and G. Szasz´ak, “Automatic assessment of spoken English
proficiency based on multimodal and multitask transformers,” in _14th_
_Intl Conf Recent Adv Natural Lang Process_, pp. 769–776, 2023.

[15] J. Pfingsthorn, _Variability in Learner Errors as a Reflection of the CLT_
_Paradigm Shift_ . Peter Lang, 2013, 2013.

[16] J. A. Hawkins and P. Buttery, “Criterial features in learner corpora:
Theory and illustrations,” _English Profile J_, vol. 1, p. e5, 2010.




[17] C. Cortes and V. Vapnik, “Support-vector networks,” _Machine learn-_
_ing_, vol. 20, no. 3, pp. 273–297, 1995.

[18] K. Zechner and I. Bejar, “Towards automatic scoring of non-native
spontaneous speech,” in _Human Lang Techn Conf NAACL_, pp. 216–
223, 2006.

[19] X. Xi, D. Higgins, K. Zechner, and D. Williamson, “Automated scoring
of spontaneous speech using speechrater,” _ETS Research Report Series_,
vol. 2008, 12 2008.

[20] Y. Wang, M. Gales, K. Knill, K. Kyriakopoulos, A. Malinin, R. van
Dalen, and M. Rashid, “Towards automatic assessment of spontaneous
spoken english,” _Speech Communication_, vol. 104, pp. 47–56, 2018.

[21] Y. Qian, K. Evanini, X. Wang, C. M. Lee, and M. D. Mulholland,
“Bidirectional lstm-rnn for improving automated assessment of nonnative children’s speech,” in _Interspeech_, 2017.

[22] Y. K. Singla, A. Gupta, S. Bagga, C. Chen, B. Krishnamurthy, and
R. R. Shah, “Speaker-conditioned hierarchical modeling for automated
speech scoring,” in _30th ACM Intl Conf Inf & Knowledge Management_,
p. 1681–1691, 2021.

[23] S. Bann`o, K. Knill, M. Matassoni, V. Raina, and M. J. F. Gales, “L2
proficiency assessment using self-supervised speech representations,”
_ArXiv_, vol. abs/2211.08849, 2022.

[24] S. Bann`o and M. Matassoni, “Proficiency assessment of l2 spoken
english using wav2vec 2.0,” in _IEEE Spoken Langu Techn Workshop_,
pp. 1088–1095, 2023.

[25] E. Islam, C. Park, and T. Hain, “Exploring speech representations
for proficiency assessment in language learning,” in _9th Workshop on_
_Speech and Language Technology in Education (SLaTE)_, pp. 151–155,
08 2023.

[26] T.-H. Lo, F.-A. Chao, T.-I. Wu, Y.-T. Sung, and B. Chen, “An effective
automated speaking assessment approach to mitigating data scarcity
and imbalanced distribution,” 2025.

[27] V. J. Schmalz and A. Brutti, “Automatic assessment of english cefr
levels using bert embeddings,” in _Italian Conference on Computational_
_Linguistics_, 2021.

[28] S. Bann`o and M. Matassoni, “Cross-corpora experiments of automatic
proficiency assessment and error detection for spoken english,” in _17th_
_Works Innov Use NLP Building Educat Appl_, (Seattle, Washington),
pp. 82–91, 2022.

[29] J.-T. Li, B.-C. Yan, T.-H. Lo, Y.-C. Wang, Y.-C. Hsu, and B. Chen,
“Automated speaking assessment of conversation tests with novel
graph-based modeling on spoken response coherence,” 2024.

[30] P. Muangkammuen and F. Fukumoto, “Multi-task learning for automated essay scoring with sentiment analysis,” in _1st Conf Asia-Pacific_
_Chapter Assoc Comput Linguistics and 10th Intl Joint Conf Nat Lang_
_Process_, (Suzhou, China), pp. 116–123, 2020.

[31] Y. Yang, J. Zhong, C. Wang, and Q. Li, “Exploring relevance and
coherence for automated text scoring using multi-task learning,” in
_34th Intl Conf Software Eng Knowledge Eng_, pp. 323–328, 2022.

[32] J. Robert, M. Webbie, _et al._, “Pydub,” 2018.

[33] H. Bredin, R. Yin, J. M. Coria, G. Gelly, P. Korshunov, M. Lavechin,
D. Fustes, H. Titeux, W. Bouaziz, and M.-P. Gill, “pyannote.audio:
neural building blocks for speaker diarization,” 2019.

[34] M. Ravanelli, T. Parcollet, P. Plantinga, A. Rouhe, S. Cornell, L. Lugosch, C. Subakan, N. Dawalatabad, A. Heba, J. Zhong, J.-C. Chou,
S.-L. Yeh, S.-W. Fu, C.-F. Liao, E. Rastorgueva, F. Grondin, W. Aris,
H. Na, Y. Gao, R. D. Mori, and Y. Bengio, “SpeechBrain: A generalpurpose speech toolkit,” 2021. arXiv:2106.04624.

[35] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and
I. Sutskever, “Robust speech recognition via large-scale weak supervision,” 2022.

[36] A. Tortel, “Anglish: Une base de donn´ees comparatives de l’anglais
r´ep´et´e et parl´e en l1 & l2,” _TIPA. Travaux interdisciplinaires sur la_
_parole et le langage_, vol. 27, pp. 111–122, 06 2008.

[37] P. Boersma and D. Weenink, “Praat: doing phonetics by computer
(version 5.1.13),” 2009.

[38] B. Bel and P. Blache, “The Resource Center for Oral Description
(CRDO),” _Interdisciplinary work of the Speech and Language Labo-_
_ratory_, vol. 25, pp. 13–18, 2006.

[39] J. Geertzen, T. Alexopoulou, and A. Korhonen, “Automatic linguistic
annotation of large scale l2 databases: The ef-cambridge open language database(efcamdat),” 2014.

[40] Z. Zhou, S. Vajjala, and S. V. Mirnezami, “Experiments on nonnative speech assessment and its consistency,” in _8th Workshop NLP_
_Computer Assisted Language Learning_, pp. 86–92, 2019.


