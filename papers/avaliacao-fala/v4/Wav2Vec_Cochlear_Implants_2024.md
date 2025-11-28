Interspeech 2024
1-5 September 2024, Kos, Greece

# **Automatic Assessment of Speech Production Skills for** **Children with Cochlear Implants Using Wav2Vec2.0 Acoustic Embeddings**


_Seonwoo Lee_ [1] _, Sunhee Kim_ [1] _, Minhwa Chung_ [1]


1Seoul National University, Republic of Korea
_{_ lsw5220,sunhkim,mchung _}_ @snu.ac.kr



**Abstract**


This study introduces an automatic assessment model for
speech production skills of children with cochlear implants
(CIs) to support home-based speech therapy. The model employs acoustic embeddings from self-supervised models and
considers speech traits of both normal hearing (NH) adults and
children, which is a novel method for evaluating speech of children with disorders. It combines phoneme embeddings and two
acoustic embeddings from Wav2Vec2.0 models, each trained on
the speech of NH adults and children, via multi-head attention.
Using a speech corpus of Korean-speaking children with CIs,
our model outperforms single-embedding methods in a Pearson correlation coefficient between predicted and expert-rated
scores, with a relative improvement of 51%. The results highlight the effectiveness of Wav2Vec2.0 acoustic embeddings and
the importance of incorporating both of typical speech patterns
of NH adults and children in assessing speech production skills
in children with CIs.

**Index Terms** : automatic assessment, speech production skills,
children with cochlear implants, self-supervised models


**1. Introduction**


Cochlear implants (CIs) is a kind of medical intervention which
can restore partial or functional hearing to individuals with hearing loss by stimulating the residual auditory nerve with electric nodes [1]. Despite improved speech perception, children
with CIs often encounter issues related to speech production
due to distorted speech perception and fixed incorrect articulation [2]. Their speech shows differences from those of children
with normal hearing (NH) in the production of segmental parts
of speech, including vowels and consonants, and in phonological process errors [3, 4]. Recent research on Korean-speaking
children with CIs has shown that although speech intelligibility
displayed no significant difference, the speech acceptability of
children with CIs was lower than that of children with NH, with
acoustic differences observed in pitch, voice quality, accentedness, resonance, and speech rate [5]. Less elaborate and stable
movements of articulators [6] are also a distinct characteristic
of these children.

To improve their speech production skills, (re)habilitation
should include continuous training which extends from therapy
sessions to the home environment where they spend most of
their time. However, parents or guardians frequently face difficulties with home-based training due to limited knowledge in
identifying errors in speech. For example, distinguishing which
phonological errors need correction is challenging, since children with CIs exhibit developmentally atypical ones compared
to children with NH as well as developmentally typical ones
which are different from the speech of adults [3, 4]. The lack



of expertise adds to their physical and mental burden. An automatic speech production assessment system could relive these
challenges associated with home training.
In contrast to relatively active research on automatic assessment systems for speech of NH children in the fields such as
reading ability assessment [7, 8] and foreign language learning [9], research on speech production assessment for children
with disorders, especially for children with CIs, has been limited. For children with speech sound disorders, decoding graphs
were extended to include mispronunciations produced by the
target children [10]. For children with hearing loss, an automatic speech production assessment system utilizing neural network output activities was proposed to identify segmental errors

[11]. Although these studies demonstrated the effectiveness of
their methods in assessing speech production skills in children,
there still remains some limitations. Primarily, only the segmental level of speech production was assessed [10, 11]. However,
diverse aspects of speech characteristics such as articulation and
prosody should be reflected to assess speech production skills
of children with CIs [5]. Furthermore, direct consideration of
speech differences between NH adults (hereafter referred to as
adults) and children with NH was absent [9, 11].

Recently, pre-trained self-supervised models have been
widely used for automatic pronunciation assessment in foreign language learning. These studies have employed acoustic embeddings extracted from fine-tuned models whose training objectives include automatic speech recognition [9, 12] or
phoneme recognition [13, 14, 15]. These embeddings are effective because they encode features related to phonetic elements,
such as consonants and vowels [16], as well as various prosodic
features, including fluency, stress, and duration [17]. Specifically, [13] employed multi-head attention to combine languagespecific embeddings, which serve as the assessment criterion
with acoustic embeddings.
In this paper, we propose an automatic speech assessment model for Korean-speaking children with CIs, leveraging
Wav2Vec2.0 acoustic embeddings. The model incorporates two
acoustic embeddings, each extracted from acoustic models finetuned with adults’ speech and NH children’s speech. Adopting language-specific embeddings representing standard speech
patterns of the native language in the foreign language learning
area, these acoustic embeddings are expected to reflect standard speech characteristics. In addition, the differences between
typical speech production characteristics of adults and children
with NH are directly considered through attention mechanisms.
The rest of the paper is organized as follows: the next section
describes the proposed method, followed by an explanation of
the experiments in Section 3. Section 4 reports experimental
results, which is analyzed and discussed in Section 5. Section 6
presents the conclusion and future work.


862 `10.21437/Interspeech.2024-234`


Figure 1: _The overview of the proposed method_


**2. Method**


The overview of the proposed method is shown in Figure 1.
The model comprises three encoders, two multi-head attentions (MHA), and a scoring module. The encoders include a
phoneme encoder, an acoustic encoder fine-tuned with adults’
speech (Acoustic Encoder-Adults; AE-A), and an acoustic encoder fine-tuned with speech from children with NH (Acoustic
Encoder-Children with NH; AE-C).

Canonical phoneme sequence and raw speech audio are input into the phoneme encoder and the acoustic encoders, respectively. The phoneme embeddings are initially processed
through an MHA with the acoustic embeddings from AE-C. The
resulting fused embeddings are then processed through another
MHA with the acoustic embeddings from AE-A. The final embeddings are passed to the scoring module, where the predicted
score is output through a convolutional layer, layer normalization, and a linear layer. The mean squared error loss function is
used for training.


**2.1. Phoneme Encoder**


The phoneme encoder takes canonical phoneme sequences as
input and outputs corresponding vectors. The phoneme embeddings are concatenated with positional embeddings, then fed
into a transformer encoder to get contextualized embeddings.
Since the length of the phoneme sequence is shorter than the
number of speech frames, a process of repeating tensors in the
contextualized embeddings is implemented to align with the dimension of acoustic embeddings. This repetition is determined
by a factor calculated by dividing the embedding dimension by
the phoneme sequence length.



**2.2. Acoustic Encoders**


Both acoustic encoders, AE-A and AE-C, are based on the
multi-lingual Wav2Vec2.0 model, XLS-R-300m [18]. XLS-R
is a large-scale model for cross-lingual speech representation
learning, pre-trained with more than 436k hours of audio data
from 128 languages. XLS-R-300m model contains over 300
millions parameters. Its architecture consists of a convolutional
feature extractor and a transformer context network. Both en
coders are fine-tuned for phoneme recognition using the Fairseq
toolkit [19]. The last hidden states of the transformer block are
extracted, then passed through a BiLSTM layer and an average
pooling layer for multi-head attention.
AE-C encodes input speech from the perspective of the
speech characteristics of children with NH. As children with
NH exhibits developmental errors or different speech production compared to adults, the resulting encoded acoustic embeddings focus on developmentally atypical speech production of
children with CIs, disregarding discrepancies related to developmental changes in children. Meanwhile, AE-A encodes input
speech from the perspective standard adult speech characteristics, representing the desired form of speech production. Thus,
the output embeddings contain information on developmentally
typical speech patterns of children with NH and developmentally atypical speech patterns of children with CIs.


**2.3. Multi-Head Attention (MHA)**


Attention is a mechanism that maps a query and a set of keyvalue pairs to an output, focusing on the similarity or compatibility [20]. MHA is employed instead of a single attention
function to jointly attend to features from different representation subspaces at different positions, capturing various relationships between the query and the key-value pairs. In the proposed model, a query generally represents the embeddings to be
processed, primarily acoustic embeddings. A key-value pair is
phoneme or acoustic embeddings to be attended to and selected
in relation to the query. As MHA focuses on the segments of a
key and a value which are most compatible with the query, the
output embeddings contain more abundant information about
the query.
The first MHA to form fused embeddings is applied between the phoneme embeddings as a key-value pair and the
acoustic embeddings from AE-C as a query. The fused embeddings encode NH children’s typical speech production matched
with canonical phoneme sequences, focusing only on atypical
patterns and disregarding developmentally typical ones. Then,
another MHA is applied with the acoustic embeddings from
AE-A as a query and the fused embedding as a key and a value
to form the final embeddings. As the acoustic embeddings from
AE-A contain information on both developmentally typical and
atypical speech patterns, the final embeddings represent acoustic information regarding typical and atypical speech errors,
with atypical ones additionally marked.


**3. Experiments**


**3.1. Dataset**


_3.1.1. Speech Corpus of Korean-Speaking Children with CIs_


A speech corpus for Korean children with CIs is utilized, which
is being constructed, as there is no available speech corpus of
children with CIs yet. All speakers in the corpus had their first
implantation before the age of three. The audio data of speech
from 30 children with CIs were extracted from videos taken dur


863


ing regular evaluation sessions for speech perception, speech
production, and language skills at Seoul National University
Hospital in Seoul, Korea. The extracted audio was segmented
for individual utterances spoken by children with CIs.
Each utterance was orthographically transcribed for the target utterance. If it was different from the target prompt during speech perception evaluation, an additional orthographic
transcription of the actual utterance was conducted. The actual utterances were then phonemically transcribed and rated
on speech production skills by three nationally certified speech
and language pathologists. Each speech and language pathologist was qualified with at least two years of experience in the
rehabilitation of children with CIs or with knowledge of phonetics. Evaluation on speech production skills was conducted
from the perspective of general speech production, considering
the general aspects of speech including articulation, prosody,
and voice quality. The score ranged from 1 to 5, with 1 indicating very poor, and 5 indicating very good speech production
skills. All processes of constructing the corpus strictly adhered
to ethical approval.
For experiments, speech from children with three years of
experience with CIs is utilized. From the initial count of 2,279
(48 minutes(m)), any utterances longer than 2 seconds are excluded to reduce length variation, resulting in a total of 1,915 utterances (29m) from 14 children with CIs. The averaged speech
production scores are used as the ground truth scores. The interrater reliability of the score assessed by the Intraclass Correlation Coefficient using SPSS version 26 is 0.733 with a 95%
confidence interval of 0.711 to 0.753 (p _<_ 0.001).


_3.1.2. Speech Corpora for Training Acoustic Encoders_


The free conversation speech corpus of Korean-speaking NH
children [21], a collection of spontaneous speech recordings of
children whose age ranges from 3 to 10, is employed to finetune AE-C. For young children with difficulties in free conversation, repeating speech is also included. The duration of the
audio amounts to more than 3,000 hours with more than 1,000
speakers. AE-C is fine-tuned with speech of NH children aged 3
to 5 in the free conversation speech corpus of Korean-speaking
children because speech data employed for the experiments are
from children with CIs who have used CI devices for three

years. The training, validation, test dataset included 256,548
(240hours(h) & 16m), 12,121 (13h & 1m), 12,000 (10h & 54m)
utterances, respectively.
KsponSpeech [22] is the dataset for fine-tuning AE-A,
which consists of Korean spontaneous speech recordings from
2,000 native Korean speakers. This corpus contains 1,000 utterances per speaker during open-domain dialog of two speakers.
Individual utterances are manually transcribed. The total duration amounts to 969 hours. For training AE-A, the training,
validation, test dataset included 620,000 (965h & 9m), 2,545(3h
& 56m), 3,000 (2h & 38m) utterances, respectively.


**3.2. Experiment Details**


The experiments are conducted via a 5-fold cross-validation approach, as the dataset of speech from children with CIs is quite
small. Each fold is randomly split into 1,532 and 383 utterances
for train and test sets, respectively. A speaker-dependent split
is employed due to the varying number of utterances per child,
which prevents imbalances in train/test data size and score distribution across folds.

For the phoneme encoder, the vocab size is 47 to include every phoneme in Korean and an extra one for space. The trans


former encoder follows the default setting of the transformer
architecture. The fine-tuning of acoustic encoders follows the
default setting of XLS-R. The best phoneme error rate (PER)
of the fine-tuned AE-C and AE-A on the test dataset achieved
8.80%, and 8.22%, respectively. The number of attention heads
in MHA is set to 8. The embedding dimension for the entire
model is set to 128. Adam optimizer is utilized with the learning rate 5e-4. Each model is trained during 50 epochs. Training a model with a fold using a GPU (NVIDIA GeForce RTX
2080 Ti) takes approximately 0.5 hours, resulting in 2.5 hours
to complete training a model.
The baseline models are models utilizing single acoustic
embeddings since there has been no study on automatic assessment of entire speech aspects of children with CIs. The experiments involve models that incorporate various combinations of
three types of embeddings: phoneme embeddings, two acoustic
embeddings from AE-C and AE-A, including the fusion of two
types of embeddings and the sequential fusion of all three types.
As the evaluation metric, Pearson correlation coefficients (PCC)
are calculated between the averaged expert-labeled scores and
predicted scores.


**4. Results**


**4.1. Experimental Results**


The experimental results are presented in Table 1 (above the
solid line). The single-acoustic embedding approaches **A** and
**C** display an average PCC of 0.4043 and 0.4983, respectively.
The fusion of acoustic embeddings **A** _⊗_ **C** and **C** _⊗_ **A** shows an
average PCC of 0.4228 and 0.4785, respectively. Attending
each acoustic embeddings with phoneme embeddings leads to a
higher PCC, 0.5599 for **P** _⊗_ **C** and 0.5645 for **P** _⊗_ **A** . The proposed method **(P** _⊗_ **C)** _⊗_ **A** achieves the best average PCC of
0.6116, with a relative improvement of 51.3% and 22.7% compared to **A** and **C**, respectively. Its standard deviation (SD)
across the folds is 0.0149, showing higher stability. The second highest PCC is obtained by **P** _⊗_ **(C** _⊗_ **A)**, with an average
PCC of 0.6075 and an SD of 0.147. The opposite configuration
**P** _⊗_ **(A** _⊗_ **C)** also displays relatively high performance compared
to the others.


**4.2. Various Combinations of Embeddings**


Since fusion of phoneme embeddings generally results in great
improvement, various combinations of the embeddings fused
with phoneme embeddings are explored as shown in Table 1
(below the solid line). Arithmetic operations involving **(P** _⊗_ **A)**
and **(P** _⊗_ **C)** enhance the PCC. Further improvement is achieved
by applying MHA.
Additional fusion is conducted to identify if using more
complex representations would result in improvement, for
example, to **(P** _⊗_ **A)** + **(P** _⊗_ **C)** (below the dashed line in Table 1). While most configurations show an increase in
the average PCC, fusion with acoustic embeddings as a key
and value decreases the performance. The configuration
**(P** _⊗_ **C)** _⊗_ **[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** yields a performance comparable to
the best performance of the proposed model, with an average
PCC of 0.6095.


**5. Discussion**


The configurations of single acoustic embeddings show a moderate correlation with an average PCC of more than 0.4, suggesting the effectiveness of Wav2Vec2.0 acoustic embeddings



864


Table 1: _Pearson correlation coefficients for predicted versus expert-labeled scores by embedding type._ _**A**_ _refers to acoustic embeddings_
_from the acoustic encoder-adults,_ _**C**_ _to acoustic embeddings from the acoustic encoder-children with normal hearing, and_ _**P**_ _to phoneme_
_embeddings. ⊗_ _denotes multi-head attention, where the former represents a set of key and value pairs, and the latter represents a query._
+ _and −_ _denotes addition and subtraction of tensors, respectively. In all cases, p-values < 0.001._


**Embeddings** **Fold 1** **Fold 2** **Fold 3** **Fold 4** **Fold 5** **Mean** **SD**


**C** 0.4578 0.4760 0.5310 0.5176 0.5091 0.4983 0.0304

**A** 0.3666 0.4314 0.3772 0.4573 0.3892 0.4043 0.0344

**A** _⊗_ **C** 0.3842 0.3810 0.4210 0.4811 0.4467 0.4228 0.0425
**C** _⊗_ **A** 0.4728 0.4329 0.4928 0.4961 0.4979 0.4785 0.0245
**P** _⊗_ **C** 0.5595 0.6047 0.5836 0.5021 0.5495 0.5599 0.0388
**P** _⊗_ **A** 0.5670 0.5800 0.5517 0.5440 0.5800 0.5645 **0.0146**
**P** _⊗_ **(A** _⊗_ **C)** 0.5757 0.5690 0.5558 0.5633 0.5979 0.5723 0.0161
**P** _⊗_ **(C** _⊗_ **A)** 0.5883 0.6089 0.5972 **0.6202** **0.6227** 0.6075 0.0147
**(A** _⊗_ **C)** _⊗_ **P** 0.4054 0.3763 0.3971 0.4796 0.3993 0.4115 0.0396
**(C** _⊗_ **A)** _⊗_ **P** 0.4648 0.4819 0.5289 0.5135 0.5108 0.5000 0.0260
**A** _⊗_ **(P** _⊗_ **C)** 0.4666 0.4191 0.3972 0.4977 0.4790 0.4519 0.0422
**C** _⊗_ **(P** _⊗_ **A)** 0.4683 0.4492 0.4848 0.5324 0.4948 0.4859 0.0312
**(P** _⊗_ **A)** _⊗_ **C** **0.6047** 0.5632 0.5282 0.5527 0.5630 0.5624 0.0276
**(P** _⊗_ **C)** _⊗_ **A** 0.5969 **0.6105** **0.6347** 0.6005 0.6155 **0.6116** 0.0149


**(P** _⊗_ **A)** + **(P** _⊗_ **C)** 0.5584 0.6062 0.6033 0.5385 0.5708 0.5754 0.0335
**(P** _⊗_ **A)** _−_ **(P** _⊗_ **C)** 0.5570 **0.6297** 0.5903 0.5319 0.5896 0.5797 0.0424
**(P** _⊗_ **C)** _−_ **(P** _⊗_ **A)** 0.5673 0.6149 0.5937 0.5188 0.5759 0.5741 0.0414
**(P** _⊗_ **C)** _⊗_ **(P** _⊗_ **A)** 0.5765 0.5787 **0.6284** 0.6077 0.6157 0.6014 0.0230
**(P** _⊗_ **A)** _⊗_ **(P** _⊗_ **C)** **0.6253** 0.5748 0.5593 0.5819 0.5994 0.5881 0.0253

**[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** _⊗_ **A** 0.5745 0.6006 0.5571 0.5761 0.6200 0.5857 0.0179

**[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** _⊗_ **C** 0.6008 0.5872 0.5491 0.6011 0.6159 0.5908 0.0245

**[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** _⊗_ **(P** _⊗_ **A)** 0.5715 0.5957 0.5629 0.5646 0.6159 0.5821 0.0151

**[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** _⊗_ **(P** _⊗_ **C)** 0.5611 0.5641 0.5742 0.5746 0.6092 0.5766 0.0069
**A** _⊗_ **[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** 0.4129 0.4070 0.4004 0.4680 0.4562 0.4289 0.0310
**C** _⊗_ **[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** 0.4258 0.4042 0.4120 0.3957 0.4513 0.4178 **0.0128**
**(P** _⊗_ **A)** _⊗_ **[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** 0.6008 0.5887 0.5310 0.5811 0.5691 0.5741 0.0307
**(P** _⊗_ **C)** _⊗_ **[(P** _⊗_ **A)** + **(P** _⊗_ **C)]** 0.5877 0.5965 0.6264 **0.6184** **0.6187** **0.6095** 0.0182



in assessing speech production skills of children with CIs, as
in previous studies on language learning [9, 12, 13, 14, 15].
Considering that the models for language learning have employed acoustic embeddings reflecting standard speech patterns
in the native language, the better performance of **C** compared
to **A** implies a more significant role of information related to
atypical speech patterns in **C** . Meanwhile, the proposed method
**(P** _⊗_ **C)** _⊗_ **A** obtains the best performance. This indicates that using acoustic embeddings that reflect the speech characteristics
of either adults or NH children alone is insufficient, emphasizing the significance of leveraging both of them.
The fusion of phoneme embeddings generally increases the
average PCC. As acoustic embeddings from self-supervised
models represent features in comprehensive aspects of speech

[16, 17], the fused embeddings would encode features beyond
the frame-level processing of the acoustic model, capturing segmental and suprasegmental characteristics related to higherlevel units, at least at the phoneme level. The improvement
in performance of the second-best **P** _⊗_ **(C** _⊗_ **A)** can be attributed
to the exclusively encoded information about both developmentally typical and atypical speech patterns, by incorporating
phonemic information into the acoustic embeddings which consider developmental speech patterns. The relatively lower performance of the opposite configuration **P** _⊗_ **(A** _⊗_ **C)** could be due
to developmental variability in NH children.
For queries, **A** seems more suitable than **C**, given that configurations involving **(C** _⊗_ **A)** or **(P** _⊗_ **C)** _⊗_ **(P** _⊗_ **A)** always perform better than their counterparts. These results suggest that
focusing on **A**, which reflect standard speech production, is crucial for the assessment as shown in [13] where language-specific
embeddings acted as assessment criterion.



The results of experiments with varying combinations of
phoneme-fused embeddings highlight the importance of using
both of the acoustic embeddings. While a configuration of an
additional MHA with **(P** _⊗_ **C)** applied to the arithmetically calculated embeddings achieves a PCC comparable to the best performance, the application of **(P** _⊗_ **A)** does not exhibit any improvement. This indicates that typical speech patterns of NH
children, or developmental speech errors reflected in the embeddings, also play an important role.


**6. Conclusion**


An automatic assessment model is proposed, for the first time,
for evaluating the speech production skills of children with CIs.
The model utilizes two acoustic embeddings from Wav2Vec2.0
acoustic encoders, each trained on speech of adults and children with NH, respectively, surpassing models relying on single
acoustic embeddings by simultaneously considering the typical
speech production patterns of both adults and NH children. Intensive experiments involving various combinations of the embeddings also underscore the importance of each acoustic embeddings and the significance of employing both embeddings.
Although the size of the dataset is relatively large for disordered speech research, the absolute size remains small, leading to a speaker-dependent approach. Therefore, future work
should include experiments with additional data once the ratings by the experts are complete. In addition, it is necessary to
explore which segmental and suprasegmental features influence
the predicted scores and to what extent. Identifying erroneous
fragments could also provide insights into how to improve children with CI’s speech production skills.



865


**7. Acknowledgements**


The process of developing the speech corpus is approved by
the Institutional Review Board (IRB) of Seoul National University (IRB No. 2203/003-006). This work was supported by
Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) [No.2022-0-00223, Development of digital therapeutics to improve communication ability of autism spectrum
disorder patients].


**8. References**


[1] F.-G. Zeng, “Trends in cochlear implants,” _Trends in Amplifica-_
_tion_, vol. 8, no. 1, pp. 1–34, 2004.


[2] Y. Park and E.-H. Jeong, “The effect of phonetic placement procedure and paired-stimulus treatment program on the alveolar
sound production in the children with cochlear implant.” _Jour-_
_nal of Speech-Language & Hearing Disorders_, vol. 16, no. 3, pp.
31–46, 2007.


[3] Y. K. Kim, M. H. Park, and D. I. Seok, “Development of articulation in pediatric user of cochlear implants,” _The Journal of Special_
_EducationTheory and Practice_, vol. 6, no. 2, pp. 67–83, 2005.


[4] J. H. Han, H. Kim, S. Y. Pae, and J. C. Shin, “A comparison
of phonological process between normal children and children
with cochlear implants,” _Communication Sciences and Disorders_,
vol. 11, no. 2, pp. 56–71, 2006.


[5] J. H. Kang and M. S. Yoon, “A comparison of the speech production ability of children with cochlear implants and children with
normal hearing,” _Journal of Speech-Language & Hearing Disor-_
_ders_, vol. 29, no. 1, pp. 13–21, 2020.


[6] N. Kim, Y. Lee, and H. S. Sim, “Articulatory characteristics as
functions of articulation distance and articulatory direction in children with cochlear implants,” _Communication Sciences and Dis-_
_orders_, vol. 27, no. 2, pp. 432–443, 2022.


[7] J. Tepperman, J. F. Silva, A. Kazemzadeh, H. You, S. Lee, A. Alwan, and S. S. Narayanan, “Pronunciation verification of children’s speech for automatic literacy assessment,” in _Proc. Inter-_
_speech 2006_ . ISCA, 2006, pp. 845–848.


[8] M. P. Black, J. Tepperman, and S. S. Narayanan, “Automatic prediction of children’s reading ability for high-level literacy assessment,” _IEEE Transactions on Audio, Speech, and Language Pro-_
_cessing_, vol. 19, no. 4, pp. 1015–1028, 2010.


[9] E. Kim, J.-J. Jeon, H. Seo, and H. Kim, “Automatic pronunciation
assessment using self-supervised speech representation learning,”
in _Proc. Interspeech 2022_ . ISCA, 2022, pp. 1411–1415.


[10] S. Dudy, S. Bedrick, M. Asgari, and A. Kain, “Automatic analysis of pronunciations for children with speech sound disorders,”
_Computer Speech & Language_, vol. 50, pp. 62–84, 2018.


[11] L. Czap, “Automated speech production assessment of hard of
hearing children,” _IEEE Journal of Selected Topics in Signal Pro-_
_cessing_, vol. 14, no. 2, pp. 380–389, 2019.


[12] F.-A. Chao, T.-H. Lo, T.-I. Wu, Y.-T. Sung, and B. Chen, “3m: An
effective multi-view, multi-granularity, and multi-aspect modeling approach to english pronunciation assessment,” in _Proc. 2022_
_Asia-Pacific Signal and Information Processing Association An-_
_nual Summit and Conference (APSIPA ASC)_, 2022, pp. 575–582.


[13] B. Lin and L. Wang, “Multi-lingual pronunciation assessment
with unified phoneme set and language-specific embeddings,” in
_Proc.2023 IEEE International Conference on Acoustics, Speech_
_and Signal Processing (ICASSP)_ . IEEE, 2023, pp. 1–5.


[14] H. Ryu, S. Kim, and M. Chung, “A Joint Model for Pronunciation
Assessment and Mispronunciation Detection and Diagnosis with
Multi-task Learning,” in _Proc. Interspeech 2023_ . ISCA, 2023,
pp. 959–963.


[15] A. I. Zahran, A. A. Fahmy, K. T. Wassif, and H. Bayomi, “Finetuning self-supervised learning models for end-to-end pronunciation scoring,” _IEEE Access_, vol. 11, pp. 112 650–112 663, 2023.




[16] T. tom Dieck, P. A. P´erez-Toro, T. Arias, E. Noeth, and P. Klumpp,
“Wav2vec behind the Scenes: How end2end Models learn Phonetics,” in _Proc. Interspeech 2022_ . ISCA, 2022, pp. 5130–5134.


[17] Y. K. Singla, J. Shah, C. Chen, and R. R. Shah, “What do audio
transformers hear? probing their representations for language delivery & structure,” in _Proc. 2022 IEEE International Conference_
_on Data Mining Workshops (ICDMW)_, 2022, pp. 910–925.


[18] A. Babu, C. Wang, A. Tjandra, K. Lakhotia, Q. Xu, N. Goyal,
K. Singh, P. von Platen, Y. Saraf, J. Pino, A. Baevski, A. Conneau,
and M. Auli, “XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale,” in _Proc. Interspeech 2022_ . ISCA,
2022, pp. 2278–2282.


[19] M. Ott, S. Edunov, A. Baevski, A. Fan, S. Gross, N. Ng, D. Grangier, and M. Auli, “fairseq: A fast, extensible toolkit for sequence modeling,” in _Proc. 2019 Conference of the North Amer-_
_ican Chapter of the Association for Computational Linguistics_
_(Demonstrations)_, June 2019.


[20] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N.
Gomez, L. Kaiser, and I. Polosukhin, “Attention is All you Need,”
in _Proc. Advances in Neural Information Processing Systems_,
I. Guyon, U. V. Luxburg, S. Bengio, H. Wallach, R. Fergus,
S. Vishwanathan, and R. Garnett, Eds., vol. 30. Curran Associates, Inc., 2017.


[21] AI-HUB, “AI-HUB free conversation speech corpus of Koreanspeaking children,” 2020, last accessed 21 February 2024.

[Online]. Available: https://www.aihub.or.kr/aihubdata/data/view.
do?currMenu=115&top _\_ Menu=100&dataSetSn=108


[22] J.-U. Bang, S. Yun, S.-H. Kim, M.-Y. Choi, M.-K. Lee, Y.-J. Kim,
D.-H. Kim, J. Park, Y.-J. Lee, and S.-H. Kim, “KsponSpeech: Korean Spontaneous Speech Corpus for Automatic Speech Recognition,” _Applied Sciences_, vol. 10, no. 19, 2020.



866


