## EvalYaks : Instruction Tuning Datasets and LoRA Fine-tuned Models for Automated Scoring of CEFR B2 Speaking Assessment Transcripts

Nicy Scaria a,c,1,∗, Silvester John Joseph Kennedyb,c,1, Thomas Latinovichb,c, Deepak
Subramani a


_aComputational and Data Science, IISc, Bangalore, India_
_bTalking Yak, Inc., Cedarburg, Wisconsin, USA_
_cTalking Yak English Learning Private Limited, Bangalore, India_


**Abstract**


Relying on human experts to evaluate CEFR speaking assessments in an e-learning environment
creates scalability challenges, as it limits how quickly and widely assessments can be conducted.
We aim to automate the evaluation of CEFR B2 English speaking assessments in e-learning environments from conversation transcripts. First, we evaluate the capability of leading open source
and commercial Large Language Models (LLMs) to score a candidate’s performance across various criteria in the CEFR B2 speaking exam in both global and India-specific contexts. Next, we
create a new expert-validated, CEFR-aligned synthetic conversational dataset with transcripts
that are rated at different assessment scores. In addition, new instruction-tuned datasets are developed from the English Vocabulary Profile (up to CEFR B2 level) and the CEFR-SP WikiAuto
datasets. Finally, using these new datasets, we perform parameter efficient instruction tuning of
Mistral Instruct 7B v0.2 to develop a family of models called _EvalYaks_ . Four models in this
family are for assessing the four sections of the CEFR B2 speaking exam, one for identifying the
CEFR level of vocabulary and generating level-specific vocabulary, and another for detecting the
CEFR level of text and generating level-specific text. _EvalYaks_ achieved an average acceptable
accuracy of 96%, a degree of variation of 0.35 levels, and performed 3 times better than the next
best model. This demonstrates that a 7B parameter LLM instruction tuned with high-quality
CEFR-aligned assessment data can effectively evaluate and score CEFR B2 English speaking
assessments, offering a promising solution for scalable, automated language proficiency evaluation.


_Keywords:_
Artificial Intelligence, CEFR B2 Speaking Assessment, Large Language Models, Grammar and
Vocabulary, Discourse Management, Interactive Communication


∗Corresponding author: nicyscaria@iisc.ac.in
_Email address:_ `nicyscaria@iisc.ac.in` [(Nicy Scaria )](https://orcid.org/0009-0004-8699-0312)
1These authors contributed equally to this work.


**1. Introduction**


The study of English is classified into Academic and Functional (General) English, each fulfilling different use cases. Academic English, prevalent in professional and educational spheres
such as universities, prioritizes a formal tone, organized writing, and precise vocabulary for tasks
such as essays, reports, and scholarly communication. In contrast, Functional English targets everyday communication skills in speaking, listening, reading, and writing, aiming for practical
application in social and personal interactions with a more informal, conversational approach

[1].
The Common European Framework of Reference for Languages (CEFR) evaluates English
proficiency on a six-level scale from A1 (Beginner) to C2 (Advanced). A CEFR B2 qualification indicates that the learner has the ability to independently use English to live, work, or study

[2]. The CEFR uses ‘can do’ descriptors to tailor teaching and assessment, aligning curriculum and educational objectives. These descriptors help educators set communicative goals and
adapt courses to specific learning needs through consultations with experts and stakeholders [3].
Learners often have a ‘spiky profile’, excelling in some language skills but struggling in others,
reflecting their Target Language Use (TLU). For example, the IELTS Life Skills Test [4] assesses only speaking and listening skills for UK visa applicants. The CEFR framework requires
adaptation to fit specific contexts and is not a universal solution. Its ‘can do’ statements define
TLUs such as Personal, Public, Occupational, and Educational, facilitating customized teaching
and assessment [5]. E-learning can create a comprehensive curriculum customized to individual
preferences and abilities, effectively using the CEFR framework to meet specific needs of the
learner and contextual demands.
Various organizations, including Cambridge English and the British Council, offer English
language programs and assessments. Cambridge English conducts in-person exams in three
categories: Schools, General and Higher Education, and Business. The General and Higher Education exams cater to career and academic requirements at five levels: A2 Key, B1 Preliminary,
B2 First, C1 Advanced, and C2 Proficiency. The B2 First exam is crucial for showcasing communication skills in English-speaking environments. On the other hand, the British Council’s
EnglishScore [6] provides a more straightforward, mobile-based online assessment for general
English users.
Our objective is to develop a range of models to automate the evaluation and scoring process
with the ability to handle the complexities of advanced language examinations such as the B2
First. Using both international and India-specific data, we strive to improve the accuracy and
relevance of assessments, particularly for Indian students, in both global and local contexts.
E-learning is uniquely positioned to address the preferences and capabilities of learners, particularly as the complexity of subjects increases [7]. It is especially beneficial for subjects that
require in-depth knowledge and detailed responses. In this context, depending on human experts to evaluate every assessment can be extremely costly and does not scale efficiently [8].
The adoption of technologies that can emulate the expertise of human assessors and automate
the evaluation process offers a feasible solution [9]. Implementing technologies that replicate
human expertise and automate evaluations could provide a solution, facilitating effective and
unbiased teaching and assessment of complex topics, thereby making quality education more
scalable and accessible.

Artificial intelligence (AI), including Generative AI (GenAI) technology, has made significant advancements in various domains, including education, healthcare, and scientific research.
GenAI models [10] can produce diverse content such as text, images, and videos using AI tech

2


niques. The integration of AI in education is increasing, albeit at a slower pace compared to other
industries, yet its potential influence on education is substantial [11]. These technologies enable
personalized educational materials, assessments, and tutoring [12]. Research [13] has indicated
that incorporating AI into educational settings has the potential to enhance students’ autonomy
in managing their own learning processes. Conversational AI, which conducts human-like conversations via text or audio based on LLMs, is widely accepted among students for task-oriented
dialogues [12]. Tools like Google Assistant have improved EFL students’ communication skills
and attitudes towards intelligent assistants in learning [14]. The integration of LLMs with educational technology is being applied to a wide range of tasks, including automated grading of short
answers [15, 16], essay scoring [17], generating questions automatically [18, 19, 20], creating
multiple-choice questions [21, 22], and developing chatbots [23]. Chatbots providing feedback
have effectively boosted vocabulary learning among Korean EFL primary students [24]. GPT4 has shown better performance over human responses in open-ended assessments at various
cognitive levels, highlighting its potential to support advanced educational evaluations [25].
For a long time, researchers have been developing automated language assessments to efficiently and accurately evaluate the abilities of English learners. SpeechRater [SM] Version 1.0
(v1.0) [26] is an automated system created to score the spontaneous speech of English learners,
and it is operationally used in the Test of English as a Foreign Language [TM] (TOEFL®) Practice Online assessment. Students find automated scoring of speaking performance and feedback
to be beneficial [27]. Recent advancements include the implementation of transformer-based
models like BERT [28] for evaluating CEFR levels of sentences, and GPT variants [29] for assessing essays written by L2 English learners [30]. These tools aim to provide unbiased, globally
applicable evaluations and focus on aligning content with recognized proficiency levels.
The primary aim of this paper is to create an automated system to evaluate and score a candidate’s performance in the CEFR B2 English speaking test, removing the need for a human
assessor and ensuring relevance in both global and Indian contexts. The framework of CEFR B2
Speaking Assessment and the automated scoring system is given in Figure 1. Secondary aims involve developing an automated system capable of identifying vocabulary and proficiency levels.
Furthermore, this model should generate CEFR B2 vocabulary and sentences. We introduced
_EvalYaks_, a set of six unique models, with four models dedicated to the primary aim and two to
the secondary. _EvalYaks_ is trained by instruction tuning Mistral Instruct 7B v0.2 using the Low
Rank Adaptation (LoRA), a parameter efficient fine tuning (PEFT) method.
Due to the scarcity of CEFR B2 English speaking assessment data specific to India, we
generated a data set of simulated candidate responses and their evaluation scores using GPT-4.
The data set was then refined with expert human feedback.
As a baseline, we first investigated the ability of leading LLMs, which are highly ranked
on the LMSYS leaderboard [31], to directly evaluate candidate responses by leveraging their
intrinsic knowledge and prompt engineering. In addition, we examined the ability of LLMs to
comprehend sentence structures and link vocabulary with CEFR proficiency levels. Our _EvalYaks_
suite of instruction fine-tuned (LoRA) models demonstrated superior performance in these tasks
compared to standard LLMs. In our experiments, for each scenario, we used two sets of evaluation instructions for the LLM: one with contextual information and one without. Standard LLMs

are not capable of satisfactory evaluation. Our findings indicate that a 7B parameter LLM, when
instruction-tuned with specific, high-quality CEFR-aligned assessment data, can be employed
for automated evaluation and scoring of CEFR B2 English speaking assessments.


3


Figure 1: Cambridge’s CEFR B2 Speaking Assessment framework and it’s automated scoring using _EvalYaks._


**2. Cambridge’s B2 First Speaking Assessment Framework**


We created _EvalYaks_ utilizing the Cambridge framework for the B2 First Exam. This speaking assessment is a component of the comprehensive exam that evaluates all aspects of language
skills, including listening, reading, writing, and speaking. The primary objective of this exam is
to assess the English language proficiency of the candidate at the CEFR B2 level. This framework
is illustrated in Figure 1 and is explained in detail in the following subsections.


_2.1. Exam format_


The speaking test involves two candidates and two examiners. The first examiner acts as
both an interlocutor and an assessor, facilitating the dialogue with the candidates. The second
examiner serves as an assessor, monitoring the interaction.
The speaking test is divided into four sections [32]. In the first section, the interlocutor
individually assesses each candidate’s social and interactional language skills. In the second


4


section, each candidate speaks at length by comparing two photographs, followed by a brief reply
from the second candidate. This section evaluates spoken production, assessing the candidate’s
ability to organize and express their thoughts coherently. In the third section, the candidates
converse based on written prompts and instructions, followed by a collaborative decision-making
task. This section assesses their pragmatic and strategic competencies as they exchange ideas,
justify opinions, and use negotiation strategies. In the final section, the candidates engage in a
discussion about the topics presented in the third section. This part assesses their capability to
delve into the subjects and substantiate their viewpoints.


**B2** **Grammar and vocabulary** **Discourse management** **Interactive communication**



**5**




- Shows a good degree of
control of a range of simple
and some complex grammatical forms.

- Uses a range of appropriate vocabulary to give and exchange views on a wide range
of familiar topics.




- Produces extended stretches of

language with very little hesitation.

- Contributions are relevant and

there is a clear organization of
ideas.

- Uses a range of cohesive devices
and discourse markers.




- Initiates and responds appropriately, linking contributions
to those of other speakers.

- Maintains and develops the
interaction and negotiates towards an outcome.



**4** Performance shares features of Bands 3 and 5.




- Initiates and responds appropriately.

- Maintains and develops the
interaction and negotiates towards an outcome with very little support.



**3**




- Shows a good degree of
control of simple grammatical
forms, and attempts some complex grammatical forms.

- Uses a range of appropriate vocabulary to give and exchange views on a range of familiar topics.




- Produces extended stretches of

language despite some hesitation.

- Contributions are relevant and

there is very little repetition.

- Uses a range of cohesive devices.



**2** Performance shares features of Bands 1 and 3.



**1**




- Shows a good degree of
control of simple grammatical
forms.

- Uses a range of appropriate
vocabulary when talking about
everyday situations.




- Produces responses which are extended beyond short phrases, despite hesitation.

- Contributions are mostly relevant,
despite some repetition.

- Uses basic cohesive devices.




- Initiates and responds appropriately.

- Keeps the interaction going
with very little prompting and
support.



Table 1: Cambridge B2 First speaking test’s assessment scales excluding metrics for pronunciation [33].


_2.2. Candidate Assessment_


The assessor and the interlocutor assess the individual performances of the candidates based
on the performance descriptors of the assessment scales [34]. In the actual assessment, the assessor does not participate in the interaction and scores the criteria based on the descriptors given in
Table 1. There are four criteria for assessment, viz., ‘grammar and vocabulary’, ‘discourse management’, ‘pronunciation’, and ‘interactive communication’. The grammar part of the ‘grammar
and vocabulary’ criteria assesses the candidate’s ability to apply grammatical rules and construct
sentences utilizing a wide range of grammatical forms. Vocabulary focuses on the candidate’s
lexical range and ability to use appropriate vocabulary in different scenarios. ‘Discourse management’ criteria assess the candidate’s aptitude to produce organized and coherent discourse
relevant to the questions/scenarios in monologues and interactions. Pronunciation focuses on the


5


candidate’s ability to produce easily comprehensible utterances. ‘Interactive communication’ assesses the candidate’s participation in a conversation, such as initiating and responding, taking
turns without hesitation, and maintaining a conversation.
In the present work, we focus on developing automated evaluators only for ‘grammar and
vocabulary’, ‘discourse management’, and ‘interactive communication’. As such, Table 1 lists
the detailed assessment scales for Cambridge’s B2 First speaking assessment excluding the pronunciation criteria.


**3.** _**EvalYaks**_ **Development Approach**


Our methodology for developing _EvalYaks_, an automated system to assess English speaking
proficiency of human test takers at the CEFR B2 level, consisted of four key stages. First, we generated simulated candidate conversation records, hereinafter referred to simply as conversation
records that replicate realistic conversations corresponding to different parts of the assessment
using the GPT-4 Turbo model (Jan’24). This data set contains conversations that mimicked actual assessment interactions between a candidate and an interlocutor or between a candidate, an
interlocutor, and a partner, as required by different parts of the assessment, along with scores
for different assessment criteria. The data were then carefully validated and aligned by experts.
We considered three assessment criteria: ‘grammar and vocabulary’, ‘discourse management’,
and ‘interactive communication’ (detailed in the rubric given in Table 1). Detailed information
on this process can be found in Section 4.1. Next, we examine the effectiveness of off-the-shelf
cutting-edge language models in evaluating these conversation records by leveraging their builtin abilities. For this assessment, we used two types of prompts: one that included contextual
data and another that did not. This stage informed us that instruction fine-tuning of models
is necessary to achieve acceptable automated evaluation systems. The third stage involved the
preparation of specialized instruction datasets to fine-tune a base language model to evaluate
conversation records. For this, we created many more conversational records for various scenarios with the corresponding scores to train the model. Each conversation record was rigorously
validated by experts. We used these conversation records and resources such as the English Vocabulary Profile (up to CEFR B2 level; [35]) and the CEFR-SP WikiAuto dataset [36] by adding
specific instructions, making them suitable for instruction-tuning. See Section 4.2 for more details. Finally, these validated data sets were utilized to create a suite of six models, one model
each for detecting and generating CEFR B2 level vocabulary ( _EvalYaks_ Vocab), and CEFR B2
sentence structures ( _EvalYaks_ CEFR), and four models ( _EvalYaks_ Parts 1-4) for automated evaluation of the performance of a candidate in the different parts of the CEFR B2 First English
speaking assessment. This structured multistage approach ensures that our system is effective
and reliable. Although we use the Cambridge CEFR B2 First exam as a reference, this approach
can be extended to other English Qualifications.


**4. Dataset Generation and Evaluation Metrics**


Educational datasets frequently contain sensitive personal information, and for particular uses
such as the CEFR B2 First English speaking evaluation, publicly available data are absent in the
Indian context. Fortunately, synthetic data can be created using contemporary LLMs. We used
OpenAI’s GPT 4 Turbo (Jan ’24) [29] to generate the synthetic dataset. This data set contains
conversation records of the CEFR B2 English speaking assessment and the corresponding evaluations. The generation process is described below.

6


_4.1. CEFR B2 English speaking assessment conversation records generation and validation_
We required a data set of the CEFR B2 English speaking assessment for Indian test takers.
We developed prompts to generate the conversation records and employed the services of subject
matter experts to validate and align the scores in the conversation records.


_4.1.1. Prompt for synthetic data generation_
A prompt acts as a directive for a language model to generate text that is not only pertinent,
but also varied, realistic, and aligned with the target application [37]. We needed to create conversations that would replicate various sections of the CEFR B2 First English speaking exam in
an Indian and global context, along with scores for the three assessment criteria. This required
the creation of a comprehensive and detailed prompt that included a wide range of information.
The prompts for the four sections of the assessment were different since each part emphasizes
different speaking skills, and the interactions differ significantly between the four parts.
Each prompt began with an overview of the particular part of the assessment. Following this,
the model was instructed to produce conversations that corresponded to specific scores for each
assessment criterion. For example, the output included assigning a score of 1 to both ‘grammar and vocabulary’ and ‘discourse management’. These directions used the Chain-of-Thought
(CoT) prompting technique [38], which guided the model through the process of generating
conversations corresponding to specific scores. After these directions, the output of the model
consisting of conversations and scores was requested in JSON format. JSON’s lightweight nature
and ease of parsing make it particularly advantageous for production servers of LLMs, enhancing
data interchange efficiency and reducing latency in API communications. Figure 2 presents an
example data set for each section of the assessment. The formats for the input and output data
were tailored according to the unique objectives of each part of the CEFR B2 English speaking

assessment.

In addition, the prompt also contained context data that incorporate vocabulary up to the
CEFR B2 level, assessment scales detailing the scoring for each criterion, and expectations were
provided. Creating an automated evaluation system customized for the Indian context necessitates the model’s ability to handle a wide range of India-specific information. Thus, Indiaspecific context details such as Indian names, locations, festivals, professions, and common hobbies throughout the country were included. This information was used to craft conversations that
blended Indian and global contexts. To ensure realistic data generation, a few-shot example strategy [39] was incorporated into the prompt utilizing example conversations from Cambridge’s B2
First exam. Furthermore, the prompt contained examples of typical questions [33] asked by the
interlocutor during the speaking portion of the exam. Appendix A has the prompts used to create
the simulated CEFR B2 English speaking assessment conversation records.


_4.1.2. CEFR B2 English speaking assessment conversation records for parts 1-4_
In parts 1, 2, and 4 of the CEFR B2 English speaking assessment, candidates have sequential
dialogues with the interlocutor. Parts 1 and 2 collect only scores for ‘grammar and vocabulary’
and ‘discourse management’, as indicated in the example data shown in Figure 2. In part 3,
candidates participate in a discussion among themselves and then interact with the interlocutor.
Part 4 provides an opportunity for the candidates to further explore the topics of Part 3 with the
interlocutor. For parts 3 and 4, the data include scores for ‘grammar and vocabulary’, ‘discourse
management’, and ‘interactive communication’.
In the data generation process, various conversations were formulated that aligned with different combinations of scores for ‘grammar and vocabulary’, and ‘discourse management’ for

7


Figure 2: Datapoint examples for different parts of the speaking assessment for instruction tuning.


parts 1 and 2. Similarly, for parts 3 and 4, distinct conversations were generated that captured a
variety of score combinations for ‘grammar and vocabulary’, ‘discourse management’, and ‘interactive communication’. The prompt instructions delineated the specific combinations to be
generated with each API request to GPT-4 Turbo. The combinations were calibrated such that
the score differences between the evaluation criteria within any part did not exceed two points,
as differences in excess of two points are extremely rare in actual assessments. It is quite improbable that a candidate would score considerably lower in ‘grammar and vocabulary’ (e.g.,
a score of one) while securing much higher scores in ‘discourse management’ and ‘interactive


8


communication’ (e.g., scores of four or five).
We developed 25 user profiles to participate in the four-part CEFR B2 English speaking
assessment for data generation. The profiles were grouped into five proficiency levels, each
containing five profiles, with an average skill level ranging from 1 to 5. For parts 1 and 2, the
groupings were determined by ‘grammar and vocabulary’ and ‘discourse management’. For
parts 3 and 4, ‘interactive communication’ was also considered. The automated evaluation of
the conversation records generated from the user profiles performed poorly with standard LLM
when the ground truth scores were in the lower range (Section 6). Consequently, we generated
more low-scoring conversation records for training _EvalYaks_ . In total, we produced 3,060 data
points (671, 667, 927, and 795 for parts 1-4, respectively). The data set for part 3 included a
broader range of scenarios due to the addition of another speaker to the conversation, leading to
more data points. Likewise, part 4 extended the variety of scenarios beyond those in parts 1 and
2.


_4.1.3. Data validation and alignment_
Given the intrinsic likelihood of errors and inconsistencies in text produced by large language
models [40], it is crucial to have subject matter experts validate the CEFR B2 English speaking
assessment conversation and their corresponding scores in the conversation records. We carried
out an alignment exercise to compare synthetically generated data with CEFR levels, enlisting
human experts. Three English as a foreign language teachers, each with over four years of
experience, assessed the quality of the generated data.
Each conversation record was subjected to separate evaluations by two out of three experts,
allowing a dual perspective on its compliance with predetermined scores and assessment goals.
The evaluations adhered to the criteria specified in Table 1, which outlines the expected outcomes
for each score within those criteria. Each expert verified how well the conversation corresponded
to the scores given for each criterion. When the conversation and the scores for each criterion
aligned, the experts marked conversation record as correctly matched. In cases where discrepancies were identified, experts recorded the specific criteria where mismatches occurred and
detailed the nature of these misalignments. Additionally, they suggested necessary modifications to the conversation to ensure alignment with the scores. These remarks were essential for
identifying the required improvements for each conversation record.
Following their individual evaluations, the experts convened to discuss their findings. This
session was pivotal in combining their insights and obtaining a consensus on the alignment of
each data point with the intended ratings. After reviewing the comments collectively and engaging in an in-depth discussion, the experts agreed on the necessary adjustments to the conversations to better align them with the assessment scales. These final revisions were made to
ensure that each conversation record accurately reflected the intended score for each criterion as
presented in Table 1. This collaborative review and revision process was crucial for improving
the overall accuracy and relevance of the data set in relation to the assessment objectives.


_4.2. Instruction dataset creation for fine-tuning EvalYaks_


_EvalYaks_ was trained utilizing the aligned conversation records to instruction tune a base
language model. In instruction tuning, input-output pairs are enhanced with explicit instructions

[41, 42, 43]. For our training process, we manually prepared three sets of instructions with varied
information [44, 45] and also incorporated paraphrased versions of these instructions to generate
additional training samples. The content of each instruction includes: _(i)_ The LLM’s designated


9


role as an evaluator of the CEFR B2 English speaking assessment and the detailed evaluation
steps; _(ii)_ The LLM’s designated role as an evaluator of the CEFR B2 English speaking assessment, the detailed evaluation steps, and performance descriptors; and _(iii)_ Evaluation steps and
an example of output format.
We merged these instructions with the conversation records, incorporating special tokens to
decipher various elements of the instruction, input, and output, to create instruction data points.
We generate 1151, 1266, 2843, and 2085 instruction data points for parts 1 through 4 to train
_EvalYaks_ . The three groups of instruction data point templates employed for training are detailed
in Appendix B.
For instruction tuning _EvalYaks_ Vocab and CEFR models, we created instruction data for the
English Vocabulary Profile (up to CEFR B2) and CEFR-SP WikiAuto datasets. The English Vocabulary Profile includes words, phrases, idioms, and collocations with different annotations of
the appropriate CEFR levels from Cambridge. We used two sets of instructions (and paraphrased
versions): one to identify and the other to generate words for specific CEFR levels, resulting in
3072 instruction data points from 5107 unique words. CEFR-SP includes 17k English sentences
annotated with CEFR levels. From this, we used 7453 data points from the WikiAuto dataset,
employing two sets of instructions (similar to the English Vocabulary Profile): identifying and
generating sentences for specific CEFR levels, resulting in 19,142 instruction data points.


_4.3. Performance metrics_


To assess the performance of _EvalYaks_ suite of models and other standard LLMs to perform
automated CEFR B2 English speaking assessment evaluation, we classified the results into four
distinct levels: accurate, partly accurate, acceptable, and inaccurate. A response from the model
is considered accurate when the scores across all assessment criteria are perfectly aligned with the
reference scores. A response is classified as partly accurate if it matches the reference score on
at least one assessment criterion. The classification of a response as acceptable involves a more
nuanced criterion: It is labeled acceptable if the scores deviate by a margin of one, in the same
direction from the reference scores, across any or all of the assessment criteria. For example,
if the reference scores are (3,3), then the response scores of (2,2), (4,4), (2,3), (3,2), (3,4) and
(4,3) are considered acceptable. However, combinations such as (2,4) or (4,2), despite both
scores being within the acceptable range, are not considered acceptable due to the inconsistent
deviation between the criteria. If any of the above conditions are not satisfied, then the response
is considered to be inaccurate, including invalid responses.
The acceptable accuracy of the model is calculated using the following formula with _Naccu_,
_Npart_, _Nacce_ and _N_ being the number of accurate, partly accurate, acceptable, and total responses
generated.

_acceptable_ ~~_a_~~ _ccuracy_ = _[N][accu]_ [ +] _[ N][part]_ [ +] _[ N][acce]_ (1)

_N_


An average acceptable accuracy is defined taking into account all four parts of the CEFR B2
English speaking assessment as


4
� _i_ =1 _[acceptable]_ ~~_[ a]_~~ _[ccuracy]_ _i_
_average_ ~~_a_~~ _cceptable_ ~~_a_~~ _ccuracy_ =, (2)

4


where _acceptable_ ~~_a_~~ _ccuracyi_ is the acceptable accuracy for each part, with _i_ = 1, 2, 3, 4 referring to the different parts of the assessment.


10


In addition to acceptable accuracy, we also calculated the degree of variation (DOV) in the
responses. DOV refers to the average variation of the assessment scores relative to its reference
value. DOV in the response for parts one and two is defined as



_DOV_ 1 = _DOV_ 2 =



_N_
�

_i_ =1



| _GVri_ − _GVai_ | + | _DMri_ − _DMai_ |

, (3)
2 _N_



and for parts three and four is defined as



_DOV_ 3 = _DOV_ 4 =



_N_
�

_i_ =1



| _GVri_ − _GVai_ | + | _DMri_ − _DMai_ | + | _ICri_ − _ICai_ |

, (4)
3 _N_



where GV stands for ‘grammar and vocabulary’, DM for ‘discourse management’, and IC for
‘interactive communication’. The subscripts ‘r’ and ‘a’ are used to differentiate between the
references and the actual scores. For DOV, smaller is better.


**5. Training** _**EvalYaks**_


Each of the six models that are part of the _EvalYaks_ suite was trained separately with mistral7b-instruct-v0.2 as the base model on which Low Rank Adaptation (LoRA) [46] was used for
parameter and memory efficient instruction fine-tuning. By introducing two smaller matrices for
weight updates via low-rank decomposition, LoRA enables adaptations to new data without altering the original weight matrix, which remains unchanged. These LoRA adapters can be merged
with the base model during inference. During training, individual adapters were developed for
each dataset, such as one adapter specifically for automated evaluation and scoring of CEFR B2
speaking assessment Part 1, another for part 2, and so on. We experimented with multiple dataset
combinations, including aggregating parts 1 and 2, parts 3 and 4, all four parts together, and all
parts in conjunction with the English Vocabulary Profile and CEFR-SP WikiAuto data sets. Upon
evaluation, it was observed that the models equipped with dedicated adapters for each individual
data set exhibited superior performance.
We experimented with combinations of hyper-parameters that control the rank of the update
matrices _r_ and the LoRA scaling factor α, such as ( _r_, α) = (64, 16), (256, 128), and (256, 512).
The best adapters were obtained with _r_ = 256 and α = 128 with a lora dropout of 0.1. In our
instruction tuning process, we trained for 5 epochs using a cosine learning rate schedule starting
at 2 _e_ [−][4], employing the AdamW optimizer [47] with a weight decay of 0.001. The adapters were
trained using bfloat16 precision on 1 or 2 NVIDIA A100 GPUs (each with 80 GB of VRAM),
3 or 4 RTX A6000 Ada GPUs (each with 48 GB of VRAM), or 3 or 4 RTX A6000 GPUs (each
with 48 GB of VRAM). This training was conducted on servers available through RunPod and
on local servers.

Thus, ‘ _EvalYaks_ ’ refers to a family of six distinct models, each resulting from the integration
of an individual adapter with the base model with the best performing hyperparameters.


**6. Results and Discussion**


We investigated the performance of _EvalYaks_ and standard LLMs without LoRA in the automated assessment of conversation records of the CEFR B2 English speaking assessment. The


11


procedure began with the evaluation of LLMs using prompts both without and with performance
descriptors. We conducted an extensive analysis of the _EvalYaks_ Vocab and CEFR models, as
well as _EvalYaks_ Part 1-4 models.
The assessment was conducted by providing the LLMs with two different sets of prompts,
which included comprehensive instructions and conversations from the test data. The first prompt
specified the LLM’s role as a CEFR B2 English speaking assessment evaluator, detailed the
evaluation steps, and included the conversation to be assessed. The second prompt included
the performance descriptors from Table 1 along with the content from the first prompt. These
descriptors served as a reference for the model to rate the assessment criteria. Both sets of
prompts used in the evaluation are presented in Appendix C.


_6.1. Performance Analysis of standard LLMs without LoRA_

We first investigated the intrinsic abilities of leading (as of March 2024) LLMs to perform
the automated CEFR B2 English speaking evaluation task. We utilized 11 different LLMs (both
proprietary and open-source of different sizes) that are at the top of the LMSYS leaderboard

[31]. These include Gemma 7B [48], Mistral Instruct v0.2 [49], Llama2 7B Chat [50], Vicuna
33B [51], Mixtral Instruct v0.1 [52], Llama2 70B Chat [50], Qwen 72B Chat [53], GPT 3.5 (Jan
’24), Claude Haiku (Mar ’24) [54], Gemini Pro 1.0 [55], and Mistral Medium. These models
vary in complexity, ranging from 7 billion parameters to some claiming to have hundreds of
billion parameters.


Model Organization Without performance With performance
descriptors descriptors


Gemini Pro 1.0 Google DeepMind **69%** **82%**
Vicuna 33B LMSYS 66% 66%

Claude Haiku (Mar ’24) Anthropic 62% 61%
Llama2 70B Chat Meta 56% 55%

Llama2 7B Chat Meta 53% 64%

Mixtral Instruct v0.1 Mistral 47% 53%

Mistral Medium Mistral 46% 80%

Qwen 72B Chat Alibaba 46% 74%
GPT 3.5 (Jan ’24) OpenAI 43% 62%
Mistral Instruct v0.2 Mistral 40% 47%

Gemma 7B Google DeepMind 29% 57%


Table 2: The average acceptable accuracy of state-of-the-art LLMs without LoRA for CEFR B2
English speaking assessment using prompts without and with performance descriptors.


Table 2 presents the average acceptable accuracy metric achieved by all evaluated models
when prompted with and without performance descriptors. It is observed that Gemini Pro 1.0
outperforms the other models, achieving an average acceptable accuracy of 69% without performance descriptors and 82% with them. Providing contextual information, such as details about
the assessment scales, clearly enhances the performance of Gemini Pro 1.0. This trend is consistent across most of the other LLMs, except for Claude haiku (Mar’24) and Llama2 70B Chat,
which saw a slight decrease of 1% in their average acceptable accuracy, and Vicuna 33B, which
showed no change. Meanwhile, the Gemma 7B, Qwen 72B Chat, and Mistral Medium models
demonstrated substantial performance improvements, with increases of 28%, 28%, and 34%,
respectively, in their average acceptable accuracy.


12


It is notable that the model’s size does not directly correlate with its capacity to assess learners’ performance in the CEFR B2 English speaking evaluation. This implies that the size of
the model does not inherently imply its proficiency for a particular task. However, adding contextual information to the prompt can greatly enhance the LLMs’ ability to evaluate learners’
performance in the CEFR B2 English speaking assessment.


_6.2. Performance analysis of EvalYaks Vocab and CEFR models_


We explored the ability of a relatively small, instruction-tuned language model to grasp the
inherent structure of sentences and link vocabulary to CEFR proficiency levels through instruction tuning. The _EvalYaks_ Vocab model, trained on a dataset created from the English Vocabulary
Profile, achieves an acceptable accuracy of 100%. The model’s accurate response rate is 85%,
while its acceptable response rate is 15%. This suggests that the model can reliably associate
vocabulary with the appropriate CEFR levels through instruction tuning.
_EvalYaks_ CEFR model, trained on an instructional dataset derived from the CEFR-SP WikiAuto dataset, achieved a satisfactory accuracy rate of 96.25% for determining the CEFR level
of sentences or generating sentences to align with specific CEFR levels. The model produced
accurate responses 65% of the time, acceptable responses 31.25% of the time, and inaccurate responses 3.75% of the time. This indicates that the model is capable of evaluating sentence-level
CEFR alignment, with most of its responses being either accurate or acceptable.


_6.3. Performance analysis of EvalYaks Part 1-4 models_


Figure 3: The average acceptable accuracy of _EvalYaks_ part 1-4 models in CEFR B2 English speaking assessment with
and without performance descriptors in the prompt.


13


Figure 4: The average acceptable accuracy of state-of-the-art LLMs without LoRA in comparison with _EvalYaks_ part 1-4
models in CEFR B2 English speaking assessment with and without performance descriptors in the prompt.


We performed an in-depth analysis to understand whether instruction tuning with highquality CEFR-aligned conversational records could improve the performance of a 7B parameter model compared to other leading LLMs without fine-tuning. Figure 3 presents the average
acceptable accuracy of the _EvalYaks_ models in different parts of the CEFR B2 English speaking assessment, both with and without the inclusion of performance descriptors in the prompts.
The _EvalYaks_ models maintain a notably high average acceptable accuracy, with 96% in part


14


Figure 5: The distribution of average acceptable accuracy of state-of-the-art LLMs without LoRA in comparison with
_EvalYaks_ part 1-4 models in CEFR B2 English speaking assessment with and without performance descriptors in the
prompt.


one when prompted with and without performance descriptors. This level of accuracy is consistent with that in part 2. In part 3, the average acceptable accuracy of the model 92% and 96%
when prompted with and without performance descriptors, respectively. In the case of part 4,
the model achieves an average acceptable accuracy of 100% with performance descriptors in the
prompt and 96% without performance descriptors in the prompt.
_EvalYaks_ models, tuned with high-quality CEFR-aligned conversation records, demonstrated
an average acceptable accuracy of 96% whether performance descriptors were included or not,
as illustrated in Figure 4. In contrast, the next best-performing model, Gemini Pro 1.0, showed

15


an average acceptable accuracy of 69% for prompts without performance descriptors and 82%
for prompts with performance descriptors. The _EvalYaks_ model family comprises 7B parameter
models, while it is speculated that Gemini Pro 1.0 has hundreds of billions of parameters. The
base model for _EvalYaks_, Mistral Instruct v0.2, achieved an acceptable accuracy rate of 40%
for prompts without performance descriptors and 47% for those with performance descriptors,
which is significantly lower than the average acceptable accuracy of _EvalYaks_ models.
We examined the models’ performance by looking at the distribution of accurate, partly accurate, acceptable, and inaccurate responses averaged across the four components of the speaking
assessment. Figure 5 shows that the _EvalYaks_ models possess the highest percentage of accurate
responses for the prompts with and without performance descriptors. The accurate response rates
are 59% for the evaluation with the prompts lacking performance descriptors and 55% for the
evaluation using the prompts with performance descriptors. In particular, Gemini Pro 1.0, which
ranked second in performance, achieved only 20% and 26% accurate responses in the evaluations
using the prompts without and with performance descriptors, respectively. The increased average acceptable accuracy in other LLMs can be attributed to an increased number of acceptable
responses rather than accurate or partly accurate ones.


_6.3.1. Analysis of EvalYaks Part 1-4 models without performance descriptors_


Figure 6: The distribution of acceptable accuracy of state-of-the-art LLMs without LoRA in comparison with _EvalYaks_
part 1-4 models using prompts without performance descriptors.


16


Figure 6 illustrates the models’ performance in terms of the distribution of accurate, partly
accurate, acceptable, and inaccurate responses in different parts of the evaluation without performance descriptors. The figure clearly shows that _EvalYaks_ models outperform others in all
sections. _EvalYaks_ part 1-4 models exhibit the highest proportion of accurate responses, with
percentages of 56%, 76%, 48%, and 56% for parts 1 to 4. In each part of the assessment, the
percentage of accurate responses exceeds that of partly accurate and acceptable responses, except
in part 3, where the percentages of accurate and acceptable responses are identical.
The analysis revealed that while standard LLMs achieved a reasonable level of acceptable
responses, their accurate and partly accurate responses were often lower. This consistent performance underscores the effectiveness of targeted instruction tuning in enhancing the accuracy and
reliability of automated evaluators for speaking assessments.


_6.3.2. Analysis of EvalYaks Part 1-4 models with performance descriptors_


Figure 7: The distribution of acceptable accuracy of state-of-the-art LLMs without LoRA in comparison with _EvalYaks_
part 1-4 models using prompts with performance descriptors.


Figure 7 demonstrates the model’s performance in the distribution of accurate, partly accurate, acceptable, and inaccurate responses in various parts of the CEFR B2 English speaking
exam when using performance descriptors. Notably, _EvalYaks_ achieves the highest accuracy rate
in all parts of the assessment, with percentages of 40%, 68%, 48%, and 56% for parts one, two,


17


three, and four, respectively. Apart from part three, _EvalYaks_ models also show a higher proportion of partly accurate responses. Although _EvalYaks_ maintain a consistent average acceptable
accuracy, their performance is reduced compared to evaluations without performance descriptors. Other models, which were not trained with CEFR-aligned conversation records, generally
exhibit a higher rate of acceptable responses compared to accurate and partly accurate responses,
with the exception of Gemma 7B and Llama2 7B in part three.
The models’ average acceptable accuracy fluctuated across various parts of the speaking
assessment. In part 1, Mistral Medium ranks second with an average acceptable accuracy of 88%.
Similarly, Gemini Pro 1.0 maintains the second position in parts 2, 3, and 4, achieving average
acceptable accuracies of 84%, 76%, and 94%, respectively. As previously mentioned, despite
the high overall average accuracy of the models, it is important to recognize that in part 1, 48%
of responses were acceptable, and in parts 2, 3, and 4, the acceptable responses were 60%, 40%,
and 48%, respectively. Upon fine-tuning with CEFR-aligned conversation records, the model
that initially scored the lowest or second lowest in different parts of the assessment exhibited
the most substantial improvement, eventually achieving the best performance in all components
of the speaking assessment. This underscores the effectiveness of specific instruction tuning in
enhancing model performance for designated tasks.


_6.3.3. Degree of variation in EvalYaks part 1-4 models_
The second performance metric used to analyze the models is the degree of variation (DOV)
in the scores assigned by the model for the input conversation in different parts of the assessment.
The equations governing DOV in scores are given in Equations 3 and 4. The DOV obtained for
different parts of the assessment when prompted without and with the performance descriptors
are given in Table 3. In the context of DOV, a lower value indicates better performance, meaning the model’s outputs are closer to the expected standards or answers. The _EvalYaks_ models
perform much better than the other models with a DOV of 0.34, 0.20, 0.52, 0.31, and 0.34 in
parts 1, 2, 3, and 4 and overall, respectively when prompted without performance descriptors.
During evaluations that incorporate performance descriptors within the prompts, the _EvalYaks_
models demonstrate consistently low DOV values, recording scores of 0.34, 0.26, 0.52, 0.31,
and 0.36 for parts 1, 2, 3, and 4, and overall, respectively. The _EvalYaks_ models perform the best
in part 2 and perform the their worse in part 3 regardless of the prompt type. Interestingly, the
performance of the model in part 2 slightly deteriorates when prompts included descriptors, giving a higher overall DOV of 0.36 compared to 0.34 in the evaluation using the prompts without
descriptors.
In contrast to the _EvalYaks_, which exhibits a slight increase in DOV when evaluated with
prompts containing performance descriptors, most other LLMs (except Claude haiku (Mar’24)
and Mixtral Instruct v0.1), which did not undergo instruction tuning with CEFR-aligned conversation records, show improved performance under the same conditions. The model ranking
next in performance, Gemini Pro 1.0, demonstrated an overall DOV of 1 when evaluated without
performance descriptors in the prompts. During the evaluation with prompts incorporating performance descriptors, Mistral Medium emerged as the second-best performing model, achieving
a DOV of 0.90. The difference in the DOV between the _EvalYaks_ models and the second-best
performing models is 0.66 for evaluation using prompts without performance descriptors and
0.54 for those with performance descriptors.
Mistral Instruct v0.2 emerges as the model with the lowest performance, recording DOV
scores of 1.65, 1.72, 1.99, 1.79, and 1.79 when assessed with prompts without performance
descriptors for parts 1, 2, 3, and 4 and overall, respectively. Conversely, when evaluated with

18


Without performance descriptors


Model Part 1 Part 2 Part 3 Part 4 Overall


_EvalYaks_ **0.34** **0.20** **0.52** **0.31** **0.34**

Gemini Pro 1.0 1.06 1.04 1.01 0.89 1.00

Vicuna 33B 1.32 1.08 1.29 1.08 1.19

Claude haiku (Mar’24) 1.18 1.26 1.36 1.08 1.22
Qwen 72B Chat 1.22 1.08 1.37 1.16 1.21
Mixtral Instruct v0.1 1.13 1.35 1.39 1.15 1.26

Llama2 70B Chat 1.25 1.28 1.39 1.33 1.31

GPT 3.5 (Jan’24) 1.22 1.44 1.35 1.27 1.32
Gemma 7B 1.26 1.46 1.51 1.43 1.36

Mistral Medium 1.42 1.37 1.44 1.23 1.36

Llama2 7B Chat 1.32 1.56 1.39 1.33 1.40

Mistral Instruct v0.2 1.65 1.72 1.99 1.79 1.79


With performance descriptors


Model Part 1 Part 2 Part 3 Part 4 Overall


_EvalYaks_ **0.34** **0.26** **0.52** **0.31** **0.36**

Gemini Pro 1.0 1.06 1.04 0.95 0.64 0.92

Vicuna 33B 1.04 0.94 1.17 1.11 1.07

Claude haiku (Mar’24) 1.24 1.16 1.37 1.13 1.23
Qwen 72B Chat 1.22 1.08 1.01 0.93 1.06
Mixtral Instruct v0.1 1.10 1.42 1.52 1.23 1.32

Llama2 70B Chat 0.92 0.88 1.29 1.01 1.03

GPT 3.5 (Jan’24) 0.88 1.02 0.96 0.93 0.99
Gemma 7B 1.26 1.02 1.28 1.17 1.18

Mistral Medium 0.78 0.92 1.01 0.88 0.90

Llama2 7B Chat 1.18 1.36 1.25 1.15 1.24

Mistral Instruct v0.2 1.30 1.34 1.89 1.37 1.48


Table 3: The degree of variation (DOV) in the scores across parts 1-4 of the assessment for
prompts without and with performance descriptors (lower values are more desirable).


prompts containing descriptors, it shows slightly improved DOV scores of 1.30, 1.34, 1.89, 1.37,
and 1.48 for parts 1, 2, 3, and 4 and overall, respectively. Despite its relatively lower performance,
it is critical to note that the _EvalYaks_ models, which were instruction tuned with CEFR-aligned
conversation using Mistral Instruct v0.2 as the base model, displayed superior performance compared to much larger models. This underscores the effectiveness of targeted instruction tuning in
improving the model’s performance for specific use cases.


**7. Conclusion**


In this study, we created _EvalYaks_, a collection of six models aimed at the automated assessment of CEFR B2 English speaking assessment parts 1 through 4, as well as CEFR Vocabulary
and sentence structures. These models were trained by instruction fine-tuning the base Mistral
Instruct V0.2 with a high-quality, well-aligned candidate conversation records dataset that we
developed. The performance of the Automated Evaluators was evaluated based on the average
acceptable accuracy and Degree of variation (DOV) in the scores.
_EvalYaks_ surpassed its primary competitor, Gemini Pro 1.0, by threefold in terms of accurate
score predictions and doubled its effectiveness when performance descriptors were included. The
base model of _EvalYaks_, Mistral Instruct V0.2, ranked near the lowest among 11 models in the

19


CEFR B2 English assessment. _EvalYaks_ consistently demonstrated strong performance in this
assessment, registering a DOV of 0.34 and 0.36 for scenarios without and with performance
descriptors, respectively. This was roughly one-third of the variation seen in Gemini Pro 1.0,
which had a DOV of 0.66 more than _EvalYaks_ . The subsequent five top models demonstrated
DOVs ranging from 0.19 to 0.31 more in contrast to Gemini Pro 1.0, indicating three to four times
greater effectiveness of _EvalYaks_ among the other leading models. Overall, _EvalYaks_ showed
consistent excellence across all evaluated metrics.
Our research confirms that large language models can effectively understand sentence structures and match vocabulary to CEFR levels. The alignment training for _EvalYaks_ models that
focused on Cambridge B2 First is applicable to the other Cambridge English Qualifications.
This adaptability suggests that similar models could be successfully used for automated CEFR
speaking assessments, offering scalable and resource-efficient solutions.
Although LoRA adapters have shown potential, they may not suffice for a robust, productionlevel system on their own. To boost reliability, we plan to incorporate _EvalYaks_ with language
agents [56], Direct Preference Optimization (DPO), and reflection [57, 58, 59] into the workflow.
The reflective agents will enable self-improvement and evaluation of the scores during the iterations preceding the final outcomes. When _EvalYaks_ produce ‘partly accurate’, ‘acceptable’, and
‘inaccurate’ responses, employing preference sampling with human experts can fine-tune these
scores. We also plan to add descriptive feedback to offer deeper insights into the performance,
highlighting their strengths and areas for improvement. This enhancement not only displays detailed scores but also elucidates the reasoning behind them, fostering learning through targeted
feedback. Furthermore, adding memory to the _EvalYaks_ can connect assessment results with
curated learning content, enriching the e-learning experience through a synergistic relationship
between assessment and educational content. As _EvalYaks_ are language models, we have used
text inputs for scoring. However, recognizing the crucial role of pronunciation in speaking assessments, a multi-modal system that can evaluate both textual and auditory elements of speech
is a future area of study.


**Acknowledgements**


This publication/presentation/research report has made use of the English Vocabulary Profile. This resource is based on extensive research using the Cambridge Learner Corpus and is
part of the English Profile programme, which aims to provide evidence about language use that
[helps to produce better language teaching materials. See http://www.englishprofile.org for more](http://www.englishprofile.org)
information.


**Appendix A. Data generation prompts**


The prompt used with GPT 4 Turbo (Jan’24) to generate synthetic conversations and assessments is given below.


_In part one, candidates engage in a one-on-one conversation with the “Examiner” covering per-_
_sonal details, routines, preferences, etc. This segment aims to evaluate their spontaneous com-_
_munication skills in everyday situations. “Candidate” should relax as the conversation starts,_
_focusing on listening attentively and providing detailed answers, avoiding mere yes or no re-_
_sponses. While they should elaborate with reasons and examples, overly lengthy responses are_
_not necessary at this stage._


20


_**Data generation instructions:**_


  - _I want to create a conversational dataset for several combinations of scores in GRAMMAR_
_AND VOCABULARY and DISCOURSE MANAGEMENT with the provided context. I need_
_the output in the JSON format as given in the ‘Data Format’._


  - _I first want to start with BAND 1 in GRAMMAR AND VOCABULARY_ & _BAND 1 in DIS-_
_COURSE MANAGEMENT._


  - _Here the conversation is between the “Examiner” and the “Candidate”. The conversation_

_lasts for 4 exchanges (4 questions from Examiner and 4 responses from the Candidate)._
_And this must be categorised as “INPUT” in the JSON._


  - _Refer to the ‘Typical Questions’ and create similar questions for the conversation_


  - _Refer to ‘Indian Context’ to create to generate Indian context specific questions and re-_
_sponses from time to time._


  - _Use the A1, A2, B1_ & _B2 CEFR vocabulary given in the context._


  - _As the OUTPUT, I want you to provide a BAND score of 1 each to Key ‘GRAMMAR_
_AND VOCABULARY’_ & _Key ‘DISCOURSE MANAGEMENT’ after reading through the_
_performance descriptors._


  - _Please review the provided “Sample Interactions” and their corresponding ratings to en-_
_sure the appropriate level of thoroughness in creating your responses._


_**Data Format:**_ { _data_ ~~_f_~~ _ormat_ }


_**Typical Questions:**_ { _typical_ ~~_q_~~ _uestions_ }


_**Performance Descriptors:**_


  - _GRAMMAR AND VOCABULARY:_ { _performance descriptor grammar and vocabulary_ }


  - _DISCOURSE MANAGEMENT:_ { _performance descriptor for discourse management_ }


_**CEFR Vocabulary:**_


  - _A1 Words:_ { _A1_ ~~_w_~~ _ords_ }


  - _A2 Words:_ { _A2_ ~~_w_~~ _ords_ }


  - _B1 Words:_ { _B1_ ~~_w_~~ _ords_ }


  - _B2 Words:_ { _B2_ ~~_w_~~ _ords_ }


_**Indian Context:**_


  - _Names:_ { _indian_ ~~_n_~~ _ames_ }


  - _Places:_ { _indian_ ~~_p_~~ _laces_ }


  - _Festivals:_ { _indian_ ~~_f_~~ _estivals_ }


  - _Professions:_ { _indian_ ~~_p_~~ _rofessions_ }


  - _Hobbies:_ { _indian_ ~~_h_~~ _obbies_ }


_**Sample Interactions:**_ { _sample_ ~~_i_~~ _nteractions_ }

21


**Appendix B. Instruction datapoint template**


The three sets of instruction datapoint for part one of the CEFR B2 English speaking assessment are given below.


_Appendix B.1. Instruction datapoint without performance descriptors_


< _s_  - _[INST]_


_**Role:**_


  - _You are an assessor of the CEFR B2 English speaking assessment. You are an expert in_
_this field with several years of experience._


  - _You will be given a conversation between an Examiner and a Candidate, and your task is to_
_give scores for two metrics for the responses given by the “Candidate” in the conversation._


_**Evaluation Steps:**_


  - _Read the conversation between the Examiner and the Candidate carefully._


  - _Assign a score for GRAMMAR AND VOCABULARY and DISCOURSE MANAGEMENT_
_on a scale of 1 to 5, where 1 is the lowest and 5 is the highest._


  - _Please disregard the response provided by “Examiner” in your evaluation._


  - _Present the assessment criteria and scores in JSON format and name it OUTPUT and_
_the OUTPUT will have two key-value pairs: GRAMMAR AND VOCABULARY, and DIS-_
_COURSE MANAGEMENT_


_**Conversation:**_ { _conversation_ }


_[_ / _INST]_


{ _output_ }


< _s_  

_Appendix B.2. Instruction datapoint with performance descriptors_


< _s_  - _[INST]_


_**Role:**_


  - _You are an assessor of the CEFR B2 English speaking assessment. You are an expert in_
_this field with several years of experience._


  - _You will be given a conversation between an Examiner and a Candidate, and your task is to_
_give scores for two metrics for the responses given by the “Candidate” in the conversation._


_**Evaluation Steps:**_


  - _Read the conversation between the Examiner and the Candidate carefully._


22


  - _Assign a score for the assessment criteria based on the ‘Performance Descriptors’ given_
_on a scale of 1 to 5, where 1 is the lowest and 5 is the highest._


  - _Please disregard the response provided by “Examiner” in your evaluation._


  - _Present the assessment criteria and scores in JSON format and name it OUTPUT and_
_the OUTPUT will have two key-value pairs: GRAMMAR AND VOCABULARY, and DIS-_
_COURSE MANAGEMENT_


_**Performance Descriptors**_


  - _GRAMMAR AND VOCABULARY:_ { _performance descriptor grammar and vocabulary_ }


  - _DISCOURSE MANAGEMENT:_ { _performance descriptor for discourse management_ }


_**Conversation:**_ { _conversation_ }


_[_ / _INST]_


{ _output_ }


< _s_  

_Appendix B.3. Instruction datapoint with evaluations steps and output format_


< _s_  - _[INST]_


_**Evaluation Steps:**_


  - _Read the conversation attentively, focusing on the interaction._


  - _Rate the Candidate’s use of GRAMMAR AND VOCABULARY on a scale from 1 to 5._


  - _Rate the Candidate’s ability in DISCOURSE MANAGEMENT on the same scale._


  - _Only assess the Candidate’s responses, ignoring the Examiner’s input._


  - _Present your evaluation scores in a JSON format named OUTPUT, with two key-value_
_pairs: GRAMMAR AND VOCABULARY and DISCOURSE MANAGEMENT._


_**Example of OUTPUT format:**_ { _output_ ~~_f_~~ _ormat_ }


_**Conversation:**_ { _conversation_ }


_[_ / _INST]_


{ _output_ }


< _s_  

**Appendix C. Evaluation prompts**


The two sets of prompts that were used to evaluate the performance of state-of-the-art LLMs
without LoRA and ‘ _EvalYaks_ ’ for part 1 are given below.


23


_Appendix C.1. Prompt without performance descriptors_


_**Role:**_


  - _You are an assessor of the CEFR B2 English speaking assessment. You are an expert in_
_this field with several years of experience._


  - _You will be given a conversation between an Examiner and a Candidate, and your task is to_
_give scores for two metrics for the responses given by the “Candidate” in the conversation._


_**Evaluation Steps:**_


  - _Read the conversation between the Examiner and the Candidate carefully._


  - _Assign a score for GRAMMAR AND VOCABULARY and DISCOURSE MANAGEMENT_
_on a scale of 1 to 5, where 1 is the lowest and 5 is the highest._


  - _Please disregard the response provided by “Examiner” in your evaluation._


  - _Present the assessment criteria and scores in JSON format and name it OUTPUT and_
_the OUTPUT will have two key-value pairs: GRAMMAR AND VOCABULARY, and DIS-_
_COURSE MANAGEMENT_


_**Conversation:**_ { _conversation_ }


_Appendix C.2. Prompt with performance descriptors_


_**Role:**_


  - _You are an assessor of the CEFR B2 English speaking assessment. You are an expert in_
_this field with several years of experience._


  - _You will be given a conversation between an Examiner and a Candidate, and your task is to_
_give scores for two metrics for the responses given by the “Candidate” in the conversation._


_**Evaluation Steps:**_


  - _Read the conversation between the Examiner and the Candidate carefully._


  - _Assign a score for the assessment criteria based on the ‘Performance Descriptors’ given_
_on a scale of 1 to 5, where 1 is the lowest and 5 is the highest._


  - _Please disregard the response provided by “Examiner” in your evaluation._


  - _Present the assessment criteria and scores in JSON format and name it OUTPUT and_
_the OUTPUT will have two key-value pairs: GRAMMAR AND VOCABULARY and DIS-_
_COURSE MANAGEMENT_


_**Performance Descriptors**_


  - _GRAMMAR AND VOCABULARY:_ { _performance descriptor grammar and vocabulary_ }


  - _DISCOURSE MANAGEMENT:_ { _performance descriptor for discourse management_ }


_**Conversation:**_ { _conversation_ }


24


**References**


[1] B. North, A. Ortega, S. Sheehan, A core inventory for general english, british council/eaquals, Retrieved August
3rd from: http://www. teachingenglish. org. uk/publications/british-council-eaquals-core-inventory-general-engli
sh (2010).

[2] Council of Europe. Council for Cultural Co-operation. Education Committee. Modern Languages Division, Common European framework of reference for languages: Learning, teaching, assessment, Cambridge University Press,
2001.

[3] Council of Europe, Council of europe (2020), common european framework of reference for
languages: Learning teaching, assessment – companion volume, [https://www.coe.int/en/web/](https://www.coe.int/en/web/common-european-framework-reference-languages)
[common-european-framework-reference-languages (2020).](https://www.coe.int/en/web/common-european-framework-reference-languages)

[[4] IELTS, Ielts life skills test, https://ielts.org/take-a-test/test-types/ielts-life-skills-test, accessed: 2024-01-22 (2017).](https://ielts.org/take-a-test/test-types/ielts-life-skills-test)

[5] University of Cambridge ESOL Examinations, Using CEFR: Principles of Good Practice, accessed: 2024-03-30
(2011).

[[6] EnglishScore. Centre For Research In English Language Learning And Assessment, Speaking test purpose and](https://resources.englishscore.com/hubfs/Security%20report%202023/Speaking%20test%20validity%20report/Validity%20Report%20-%20Speaking%20(2).pdf)
[content, Validity report, EnglishScore, accessed: 2023-12-04 (2023).](https://resources.englishscore.com/hubfs/Security%20report%202023/Speaking%20test%20validity%20report/Validity%20Report%20-%20Speaking%20(2).pdf)
URL [https://resources.englishscore.com/hubfs/Security%20report%202023/Speaking%20test%20validity%](https://resources.englishscore.com/hubfs/Security%20report%202023/Speaking%20test%20validity%20report/Validity%20Report%20-%20Speaking%20(2).pdf)
[20report/Validity%20Report%20-%20Speaking%20(2).pdf](https://resources.englishscore.com/hubfs/Security%20report%202023/Speaking%20test%20validity%20report/Validity%20Report%20-%20Speaking%20(2).pdf)

[7] D. Morris, Economies of scale and scope in e-learning, Studies in higher education 33 (3) (2008) 331–343.

[8] N. B. Shah, J. Bradley, S. Balakrishnan, A. Parekh, K. Ramchandran, M. J. Wainwright, Some scaling laws for
mooc assessments, in: KDD Workshop on Data Mining for Educational Assessment and Feedback (ASSESS
2014), 2014.

[9] I. Mekterovi´c, L. Brki´c, M. Horvat, Scaling automated programming assessment systems, Electronics 12 (4) (2023)
942.

[10] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, Y. Bengio, Generative
adversarial nets, Advances in neural information processing systems 27 (2014).

[11] E. Kasneci, K. Seßler, S. K¨uchemann, M. Bannert, D. Dementieva, F. Fischer, U. Gasser, G. Groh, S. G¨unnemann,
E. H¨ullermeier, et al., Chatgpt for good? on opportunities and challenges of large language models for education,
Learning and individual differences 103 (2023) 102274.

[12] H. Ji, I. Han, Y. Ko, A systematic review of conversational ai in language education: Focusing on the collaboration
with human teachers, Journal of Research on Technology in Education 55 (1) (2023) 48–63.

[[13] A. Darvishi, H. Khosravi, S. Sadiq, D. Gaˇsevi´c, G. Siemens, Impact of ai assistance on student agency, Computers](https://www.sciencedirect.com/science/article/pii/S0360131523002440)
& Education 210 (2024) 104967. `[doi:https://doi.org/10.1016/j.compedu.2023.104967](https://doi.org/https://doi.org/10.1016/j.compedu.2023.104967)` .
[URL https://www.sciencedirect.com/science/article/pii/S0360131523002440](https://www.sciencedirect.com/science/article/pii/S0360131523002440)

[14] T.-Y. Tai, H. H.-J. Chen, The impact of google assistant on adolescent efl learners’ willingness to communicate,
Interactive Learning Environments 31 (3) (2023) 1485–1502.

[15] J. Schneider, B. Schenk, C. Niklaus, Towards llm-based autograding for short textual answers, arXiv preprint
arXiv:2309.11508 (2023).

[16] G. Kortemeyer, Performance of the pre-trained large language model gpt-4 on automated short answer grading,
arXiv preprint arXiv:2309.09338 (2023).

[[17] M. Stahl, L. Biermann, A. Nehring, H. Wachsmuth, Exploring LLM prompting strategies for joint essay scoring](https://aclanthology.org/2024.bea-1.23)
[and feedback generation, in: E. Kochmar, M. Bexte, J. Burstein, A. Horbach, R. Laarmann-Quante, A. Tack,](https://aclanthology.org/2024.bea-1.23)
V. Yaneva, Z. Yuan (Eds.), Proceedings of the 19th Workshop on Innovative Use of NLP for Building Educational
Applications (BEA 2024), Association for Computational Linguistics, Mexico City, Mexico, 2024, pp. 283–298.
[URL https://aclanthology.org/2024.bea-1.23](https://aclanthology.org/2024.bea-1.23)

[18] N. Mulla, P. Gharpure, Automatic question generation: a review of methodologies, datasets, evaluation metrics,
and applications, Progress in Artificial Intelligence 12 (1) (2023) 1–32.

[19] N. Scaria, S. Dharani Chenna, D. Subramani, Automated educational question generation at different bloom’s
skill levels using large language models: Strategies and evaluation, in: International Conference on Artificial
Intelligence in Education, Springer, 2024, pp. 165–179.

[20] N. Scaria, S. Chenna, D. Subramani, How good are modern llms in generating relevant and high-quality questions
at different bloom’s skill levels for indian high school social science curriculum?, in: Proceedings of the 19th
Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2024), 2024, pp. 1–10.

[21] W. Feng, J. Lee, H. McNichols, A. Scarlatos, D. Smith, S. Woodhead, N. Ornelas, A. Lan, Exploring automated
distractor generation for math multiple-choice questions via large language models, in: Findings of the Association
for Computational Linguistics: NAACL 2024, 2024, pp. 3067–3082.

[22] J. Lee, D. Smith, S. Woodhead, A. Lan, Math multiple choice question generation via human-large language model
collaboration, arXiv preprint arXiv:2405.00864 (2024).


25


[23] Y. Dan, Z. Lei, Y. Gu, Y. Li, J. Yin, J. Lin, L. Ye, Z. Tie, Y. Zhou, Y. Wang, et al., Educhat: A large-scale language
model-based chatbot system for intelligent education, arXiv preprint arXiv:2308.02773 (2023).

[24] J. Jeon, Chatbot-assisted dynamic assessment (ca-da) for l2 vocabulary learning and diagnosis, Computer Assisted
Language Learning 36 (7) (2023) 1338–1364.

[25] L. Rodrigues, F. D. Pereira, L. Cabral, D. Gaˇsevi´c, G. Ramalho, R. F. Mello, Assessing the quality of automaticgenerated short answers using gpt-4, Computers and Education: Artificial Intelligence 7 (2024) 100248.

[26] X. Xi, D. Higgins, K. Zechner, D. M. Williamson, Automated scoring of spontaneous speech using speechratersm
v1. 0, ETS Research Report Series 2008 (2) (2008) i–102.

[27] L. Gu, L. Davis, J. Tao, K. Zechner, Using spoken language technology for generating feedback to prepare for
the toefl ibt® test: A user perception study, Assessment in Education: Principles, Policy & Practice 28 (1) (2021)
58–76.

[28] T. Rama, S. Vajjala, Are pre-trained text representations useful for multilingual and multi-dimensional language
proficiency modeling?, arXiv preprint arXiv:2102.12971 (2021).

[29] J. Achiam, S. Adler, S. Agarwal, L. Ahmad, I. Akkaya, F. L. Aleman, D. Almeida, J. Altenschmidt, S. Altman,
S. Anadkat, et al., Gpt-4 technical report, arXiv preprint arXiv:2303.08774 (2023).

[30] K. P. Yancey, G. Laflair, A. Verardi, J. Burstein, Rating short l2 essays on the cefr scale with gpt-4, in: Proceedings
of the 18th Workshop on Innovative Use of NLP for Building Educational Applications (BEA 2023), 2023, pp.
576–584.

[31] L. Zheng, W.-L. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li, D. Li, E. Xing, et al., Judging
llm-as-a-judge with mt-bench and chatbot arena, Advances in Neural Information Processing Systems 36 (2024).

[32] L. Taylor, The cambridge approach to speaking assessment, Research Notes 13 (2003) 2–4.

[33] Cambridge, B2 first handbook for teachers for exams, [https://www.cambridgeenglish.org/images/](https://www.cambridgeenglish.org/images/167791-b2-first-handbook.pdf)
[167791-b2-first-handbook.pdf (2023).](https://www.cambridgeenglish.org/images/167791-b2-first-handbook.pdf)

[34] A. Ffrench, The development of a set of assessment criteria for speaking tests, Research Notes 13 (2003) 8–16.

[35] T. Alexopoulou, Building new corpora for english profile, Research Notes 33 (2008) 15–19.

[36] Y. Arase, S. Uchida, T. Kajiwara, Cefr-based sentence difficulty annotation and assessment, in: Proceedings of the
2022 Conference on Empirical Methods in Natural Language Processing, 2022, pp. 6206–6219.

[37] D. Chen, C. Lee, Y. Lu, D. Rosati, Z. Yu, Mixture of soft prompts for controllable data generation, in: Findings of
the Association for Computational Linguistics: EMNLP 2023, 2023, pp. 14815–14833.

[38] J. Wei, X. Wang, D. Schuurmans, M. Bosma, F. Xia, E. Chi, Q. V. Le, D. Zhou, et al., Chain-of-thought prompting
elicits reasoning in large language models, Advances in neural information processing systems 35 (2022) 24824–
24837.

[39] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry,
A. Askell, et al., Language models are few-shot learners, Advances in neural information processing systems 33
(2020) 1877–1901.

[40] Z. Ji, N. Lee, R. Frieske, T. Yu, D. Su, Y. Xu, E. Ishii, Y. J. Bang, A. Madotto, P. Fung, Survey of hallucination in
natural language generation, ACM Computing Surveys 55 (12) (2023) 1–38.

[41] V. Sanh, A. Webson, C. Raffel, S. Bach, L. Sutawika, Z. Alyafeai, A. Chaffin, A. Stiegler, A. Raja, M. Dey,
et al., Multitask prompted training enables zero-shot task generalization, in: International Conference on Learning
Representations, 2021.

[42] S. Mishra, D. Khashabi, C. Baral, H. Hajishirzi, Cross-task generalization via natural language crowdsourcing
instructions, in: ACL, 2022.

[43] Y. Wang, S. Mishra, P. Alipoormolabashi, Y. Kordi, A. Mirzaei, A. Arunkumar, A. Ashok, A. S. Dhanasekaran,
A. Naik, D. Stap, et al., Super-naturalinstructions:generalization via declarative instructions on 1600+ tasks, in:
EMNLP, 2022.

[44] C. Zhou, P. Liu, P. Xu, S. Iyer, J. Sun, Y. Mao, X. Ma, A. Efrat, P. Yu, L. Yu, et al., Lima: Less is more for
alignment, Advances in Neural Information Processing Systems 36 (2024).

[45] R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, T. B. Hashimoto, Alpaca: A strong,
[replicable instruction-following model, Stanford Center for Research on Foundation Models. https://crfm.stanford.](https://crfm.stanford. edu/2023/03/13/alpaca.html)
[edu/2023/03/13/alpaca.html 3 (6) (2023) 7.](https://crfm.stanford. edu/2023/03/13/alpaca.html)

[[46] E. J. Hu, yelong shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, W. Chen, LoRA: Low-rank adaptation of](https://openreview.net/forum?id=nZeVKeeFYf9)
[large language models, in: International Conference on Learning Representations, 2022.](https://openreview.net/forum?id=nZeVKeeFYf9)
[URL https://openreview.net/forum?id=nZeVKeeFYf9](https://openreview.net/forum?id=nZeVKeeFYf9)

[47] I. Loshchilov, F. Hutter, Decoupled weight decay regularization, in: International Conference on Learning Representations, 2018.

[48] G. Team, T. Mesnard, C. Hardin, R. Dadashi, S. Bhupatiraju, S. Pathak, L. Sifre, M. Rivi`ere, M. S. Kale, J. Love,
et al., Gemma: Open models based on gemini research and technology, arXiv preprint arXiv:2403.08295 (2024).

[49] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. d. l. Casas, F. Bressand, G. Lengyel,
G. Lample, L. Saulnier, et al., Mistral 7b, arXiv preprint arXiv:2310.06825 (2023).


26


[50] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al., Llama 2: Open foundation and fine-tuned chat models, arXiv preprint arXiv:2307.09288 (2023).

[51] Z. Zhang, C. Zheng, D. Tang, K. Sun, Y. Ma, Y. Bu, X. Zhou, L. Zhao, Balancing specialized and general skills in
llms: The impact of modern tuning and data strategy, arXiv preprint arXiv:2310.04945 (2023).

[52] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas, E. B.
Hanna, F. Bressand, et al., Mixtral of experts, arXiv preprint arXiv:2401.04088 (2024).

[53] J. Bai, S. Bai, Y. Chu, Z. Cui, K. Dang, X. Deng, Y. Fan, W. Ge, Y. Han, F. Huang, et al., Qwen technical report,
arXiv preprint arXiv:2309.16609 (2023).

[[54] Anthropic, Introducing the next generation of claude, https://www.anthropic.com/news/claude-3-family (2024).](https://www.anthropic.com/news/claude-3-family)

[55] G. Team, R. Anil, S. Borgeaud, Y. Wu, J.-B. Alayrac, J. Yu, R. Soricut, J. Schalkwyk, A. M. Dai, A. Hauth, et al.,
Gemini: a family of highly capable multimodal models, arXiv preprint arXiv:2312.11805 (2023).

[56] L. Wang, C. Ma, X. Feng, Z. Zhang, H. Yang, J. Zhang, Z. Chen, J. Tang, X. Chen, Y. Lin, et al., A survey on large
language model based autonomous agents, Frontiers of Computer Science 18 (6) (2024) 1–26.

[57] A. Madaan, N. Tandon, P. Gupta, S. Hallinan, L. Gao, S. Wiegreffe, U. Alon, N. Dziri, S. Prabhumoye, Y. Yang,
et al., Self-refine: Iterative refinement with self-feedback, Advances in Neural Information Processing Systems 36
(2024).

[58] N. Shinn, F. Cassano, A. Gopinath, K. Narasimhan, S. Yao, Reflexion: Language agents with verbal reinforcement
learning, Advances in Neural Information Processing Systems 36 (2024).

[59] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, C. Finn, Direct preference optimization: Your
language model is secretly a reward model, Advances in Neural Information Processing Systems 36 (2024).


27


