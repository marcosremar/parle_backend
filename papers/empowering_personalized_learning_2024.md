## **Empowering Personalized Learning through a** **Conversation-based Tutoring System with Student Modeling**



[Minju Park](https://orcid.org/0000-0002-2588-2674)
minju.park@riiid.co
Riiid AI Research

Seoul, Republic of Korea



[Sojung Kim](https://orcid.org/0000-0003-4585-3587)
sojung.kim@riiid.co
Riiid AI Research

Seoul, Republic of Korea



[Seunghyun Lee](https://orcid.org/0009-0004-2082-9091)
seunghyun.lee@riiid.co
Riiid AI Research

Seoul, Republic of Korea



[Soonwoo Kwon](https://orcid.org/0000-0003-1764-3128)

soonwoo.kwon@riiid.co
Riiid AI Research

Seoul, Republic of Korea


**ABSTRACT**


As the recent Large Language Models(LLM’s) become increasingly
competent in zero-shot and few-shot reasoning across various domains, educators are showing a growing interest in leveraging these
LLM’s in conversation-based tutoring systems. However, building
a conversation-based personalized tutoring system poses considerable challenges in accurately assessing the student and strategically
incorporating the assessment into teaching within the conversation. In this paper, we discuss design considerations for a personalized tutoring system that involves the following two key components: (1) a student modeling with diagnostic components, and (2)
a conversation-based tutor utilizing LLM with prompt engineering that incorporates student assessment outcomes and various
instructional strategies. Based on these design considerations, we
created a proof-of-concept tutoring system focused on personalization and tested it with 20 participants. The results substantiate that
our system’s framework facilitates personalization, with particular
emphasis on the elements constituting student modeling. A web
[demo of our system is available at http://rlearning-its.com.](http://rlearning-its.com)


**KEYWORDS**


Intelligent Tutoring System, Student Modeling, Personalized Learning


**ACM Reference Format:**

Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok
Kim. 2024. Empowering Personalized Learning through a Conversationbased Tutoring System with Student Modeling. In _Extended Abstracts of_
_the CHI Conference on Human Factors in Computing Systems (CHI EA ’24),_
_May 11–16, 2024, Honolulu, HI, USA._ ACM, New York, NY, USA, 10 pages.
[https://doi.org/10.1145/3613905.3651122](https://doi.org/10.1145/3613905.3651122)


**1** **INTRODUCTION**


Large Language Models (LLM’s) such as GPT [2, 24] and PaLM [5]are
excelling in diverse tasks, including common sense and knowledge


Permission to make digital or hard copies of part or all of this work for personal or
classroom use is granted without fee provided that copies are not made or distributed
for profit or commercial advantage and that copies bear this notice and the full citation
on the first page. Copyrights for third-party components of this work must be honored.
For all other uses, contact the owner/author(s).
_CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA_
© 2024 Copyright held by the owner/author(s).
ACM ISBN 979-8-4007-0331-7/24/05.
[https://doi.org/10.1145/3613905.3651122](https://doi.org/10.1145/3613905.3651122)



[Kyuseok Kim](https://orcid.org/0009-0004-2369-6335)
kimkyu80@gmail.com
Riiid AI Research

Seoul, Republic of Korea


reasoning, to mathematical, symbolic and visuo-linguistic tasks.
Surprisingly, these models are shown to demonstrate zero-shot and
few-shot reasoning capacities somewhat akin to general artificial
intelligence [3], despite being trained with a simple autoregressive
text generation objective.
Accordingly, a growing number of educators are showing interest in leveraging these LLM’s to design an all-time available,
personalized tutoring system [1] [18, 21]. For example, prior works
have developed LLM-driven conversation-based tutors in various
skill domains, including argumentation [32], reading comprehension [16], mathematical reasoning [20], and language learning [27].
However, most prior works lack a systematic student assessment,
relying completely on text inputs implicitly. Such absence of diagnostic capabilities may limit the level of achievable personalization.
To address the aforementioned issue, we discuss the design considerations for a conversation-based personalized tutoring system.
Specifically, we first focus on the diagnostic components of the
tutoring system, where these components allow the LLM to incorporate multi-faceted student assessment results via prompting. We
consider three assessment criteria: cognitive state, affective state,
and learning style.
Next, we focus on implementing personalized instructional strategies, where the LLM is prompted to provide user-tailored tutoring by incorporating diagnostic results. Specifically, we equip the
system with various personalized instructional strategies such as
adaptive exercise selection, interventional messages tailored to individual cognitive and affective states, and teaching customized to
learning style. Overall, the resulting design is a versatile personalized tutoring system that utilizes a LLM in a plug-and-play fashion.
We believe such system can be implemented across various skill
domains through the use of domain-specific prompts.
Building upon these design considerations, we introduce our
proof-of-concept implementation of a personalized tutoring system. [2] Our system teaches three English writing concepts of “Pronouns”, “Punctuation”, and “Transitions” from the SAT Writing test.
As depicted in Figure 1, the user progresses through an onboarding
survey, pre-test, tutoring session, and post-test for each concept,


[1See also https://www.khanacademy.org/ and https://www.duolingo.com/ for product](https://www.khanacademy.org/)
examples of interactive, personalized education platforms utilizing LLM’s.
2For readers interested in experiencing the system firsthand, a web demo is accessible
[at http://rlearning-its.com.](http://rlearning-its.com)


CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim



**Figure 1: Overview of our proof-of-concept personalized tu-**
**toring system**


with subsequent concepts following the same pattern. Throughout the process, the student states undergo assessment and are
subsequently applied to the main tutoring session.
In order to ascertain the system’s ability in implementing personalization within the tutoring process, we executed tutoring sessions
with the participation of 20 individuals. We provide comprehensive
instances of how our system applies personalization in student
modeling. Additionally, we offer insights and discussions stemming
from the results.

We highlight the main contributions of our paper below:


  - We discuss important design considerations for a personalized tutoring system that involves (1) diagnostic components
that support student assessment; and (2) a conversationbased tutor utilizing LLM prompted to incorporate student
assessment and instructional strategies into teaching.

  - Based on the design considerations, we implement a simple
proof-of-concept tutoring system.

  - We closely examined interactions between the system and
actual learners to assess its alignment with our design objectives. This process provided critical insights into potential
improvements and identified specific challenges, thereby
inviting the HCI community to address these issues and
advance the field of personalized educational technology.


**2** **RELATED WORK**

**2.1** **Intelligent Tutoring Systems**


Intelligent Tutoring Systems (ITS) have been a subject of extensive
research for providing immediate, personalized one-to-one tutoring experiences to learners through computer-based educational
technologies [22, 23]. As LLM’s such as GPT and PaLM have revolutionized the landscape of natural language generation, it has also
extended its potential to facilitate natural language-based ITS [13],
which serves as an AI-equipped tutoring agent that engages in
dialogues with the students. Such integration of natural language
interface holds the potential to enhance the personalization, interactivity, and adaptability of learning experiences within the context
of ITS.



Recent advances in conversation-based tutoring systems utilizing LLM’s have showcased their transformative potential [18]. For
instance, ArgueTutor [32] offers adaptive tutoring for argumentation skills, DIRECT [16] specializes in reading comprehension
tutoring systems, MATHDIAL [20] focuses on math reasoning problems, and the work by [27] enhances language learning experiences.
Moreover, prompt engineering-based tutoring systems have broadened the horizons of possibilities [4, 14].
However, the mentioned tutoring systems lack the explicit incorporation of student modeling, relying instead on text-based cues
to track student progress. The potential downside to this approach
is that these systems may struggle with insufficient diagnostic capabilities resulting from the gap in explicit student modeling. This
absence of diagnostic capabilities can significantly impact the level
of achievable personalization, particularly in terms of selecting
adaptive tutoring strategies based on student assessment. We expect this limitation to be especially profound for a tutoring system
that completely relies on in-context learning (zero- or few-shot) of
LLMs, due to lack of task-specific fine-tuning.
Addressing such limitations requires the integration of extensive student modeling from various perspectives, e.g., cognitive
and affective states, learning style, etc. Moreover, the development
of LLM prompt-based systems should consider incorporating a
system structure where prompts are adaptively updated based on
student modeling. This dual approach of refining student modeling and designing user-adaptive prompts for conversation-based
tutoring system can collectively contribute to the adaptability and
personalization of the learning process.


**2.2** **Cognitive Diagnostic Modeling**


Cognitive Diagnosis Modeling (CDM) refers to the task of assessing
a given examinee’s knowledge proficiency based on test results.
Here, a test is usually comprised of questions whose correctness can
be modeled by a binary random variable. Since accurate assessment
can result in the efficient selection of learning materials, CDM’s
are often employed for item recommendations within e-learning
environments, including ITS [34].
The most influential, traditional methods for CDM include Item
Response Theory (IRT; [10]) and DINA [7]. Within this framework, we intend to employ CDM to assess students using the IRT
model. The outcome of the assessments would subsequently provide adaptive recommendations for exercises, thereby enhancing
the personalization of our system.


**3** **DESIGN OF PERSONALIZED TUTORING**

**SYSTEM**


In this section, we discuss design considerations for a personalized
tutoring system that involves (1) multi-faceted student assessment,
and (2) LLM prompt-based tutoring that incorporates student assessment outcomes and various instructional strategies.
To aid in understanding the context of these considerations, we
first outline the user flow that has been designed to optimize their
application within our entire approach. The flow begins with an onboarding survey that introduces users to the system’s features and
establishes their initial student profile. They then take a 15-question


Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA



pre-test on SAT Writing concepts, followed by personalized tutoring based on the pre-test results. After tutoring, students take a
post-test to gauge their improvement, and this iterative process
continues until all three knowledge concepts are covered. For more
in-depth information regarding the specifics of this user flow and its
alignment with our design considerations, readers are encouraged
to refer to the Appendix A.


**3.1** **Student Assessment**


We first address student assessment criteria of our student modeling. They are strategically chosen not only for their pedagogical
significance but also based on the capabilities of LLM, particularly
GPT-4, to understand and operationalize these criteria. The system relies on three fundamental assessment criteria to shape its
instructional strategies: (1) cognitive state, (2) affective state, and (3)
learning style. Each criterion is discussed in detail below. For every
criterion, we provide a definition, measurement methodology, and
relevant learning science literature that underpins our approach.


_3.1.1_ _Cognitive State._ A cognitive state encompasses various statuses influencing an individual’s abilities in learning, comprehension, problem-solving, and decision-making. This includes, but is
not limited to, factors such as attention, memory, problem-solving
abilities, proficiency levels in particular knowledge domains, and
metacognition. In the context of this study, we narrow our focus
to those aspects of cognitive state that can be empirically assessed
using our system, which incorporates both the CDM and GPT-4.
Specifically, we examine:


  - Proficiency levels in knowledge concepts, as quantified by
IRT;

  - Metacognition, as captured through self-reported data on
students’ self-assessment of their grades and proficiency,
which offers insights into their self-regulatory practices and
self-awareness;

  - GPT-4’s session-end summary, which provides an evaluation
of a student’s cognitive state based on their interactions
during the tutoring session, and;

  - Learning Gain, as measured by the difference in proficiency
between pre-test and post-test, facilitating insights into student progress as well as engagement levels, and shaping
action items for subsequent tutoring sessions.


The importance of considering a student’s cognitive state in
personalized learning is well-documented in educational research.
Proficiency levels in knowledge concepts are crucial for selecting
learning materials that fall within the student’s Zone of Proximal
Development, optimizing the effectiveness of the educational experience [17]. Furthermore, metacognitive knowledge and learning
strategies have been found to influence learning outcomes independently of intelligence [19, 26, 30].


_3.1.2_ _Affective State._ Affective state refers to the emotional and
attitudinal aspects that influence a learner’s engagement with their
overall learning experience, encompassing elements such as motivation, interest, and self-efficacy. In this study, we aim to quantify
the affective state through the following means, acknowledging
that our diagnosis of the affective state is inherently limited.




  - The discrepancy between proficiency levels determined by
IRT models and those self-reported by the students. This
metric helps identify gaps between perceived and actual skill
levels, which can serve as indicators of emotional factors
like self-efficacy, anxiety, or overconfidence.

  - GPT-4’s session-end summary, which encapsulates key affective indicators including engagement level, motivation, and
favorability toward the subject matter. These indicators are
inferred from student dialogues and interactions that occur
during the tutoring sessions.


Prior research has shown that emotional and attitudinal factors

not only act as reliable predictors of academic success but also have
a profound impact on the efficacy of teaching methodologies [6, 25,
28, 33].


_3.1.3_ _Learning Style._ Learning style denotes the preferred means
by which an individual absorbs, processes, comprehends, and retains information [9, 11, 12]. We adopt the Felder and Silverman
learning style model [12], which classifies learning styles into 16
categories, depending on how one prefers the modes of: Perception
(sensory vs intuitive), Input (visual vs auditory), Processing (active vs reflective), and Understanding (sequential vs global). As our
system relies on a chat-based interface, the adaptation of teaching
methods within the Input mode was constrained. Therefore, our
system takes into consideration three distinct modes: Perception,
Processing, and Understanding.
In our system, learning styles are identified by:


  - Self-reported learning style, where students explicitly indicate their preferences across the three dimensions based
on provided explanations. This data sets the initial conditions for personalized instruction, offering an individualized
starting point for each student’s educational journey.

  - GPT-4’s session-end summary, which tries to discern the
learning styles demonstrated during the tutoring sessions by
examining modes of interaction and question-answering patterns. Such insights are then used for continual adjustments
in learning strategies, supplemented with comments.


Numerous studies suggest that understanding a student’s learning style can profoundly enhance the adaptation of learning and
teaching methods [1, 8, 9, 15]. We prioritize learning style as it
provides a clear framework to refine our prompting algorithms,
further individualizing the teaching experience.
The described student modeling will underpin the creation and
design of the personalized tutoring that we’ll delve into in the following section. We have designed GPT-4’s session-end summary
to incorporate specific action items related to instructional strategies based on the assessed status of the student. This integration is
intentional, aiming to directly and explicitly feed strategic actions
into GPT-4’s system prompt for more targeted tutoring.


**3.2** **LLM Prompt-Based Personalized Tutoring**
**System**


In this section, we discuss the personalization strategies we have implemented in our proof-of-concept tutoring system. With this user
flow as our foundation, we then explore how to employ cognitive


CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim


**Figure 2: Cyclical framework and structure of the system prompt in our proof-of-concept personalized tutoring system**



diagnostic modeling, namely IRT, and leverage the specific capabilities of GPT-4 in our proof-of-concept system to design prompts. The
overall structure of prompt design within our system is illustrated
in Figure 2.


_3.2.1_ _Adaptive Exercise Selection._ The first strategy we consider is
the adaptive selection of learning materials for tutoring. We used
the one-dimensional 2PL IRT model for each knowledge component,
where the probability of a correct response is modeled by:


1
_𝑃_ (correct| _𝑎,𝑑,𝜃_ ) = 1 + exp(− _𝑎_      - ( _𝜃_ − _𝑑_ )) _[,]_ (1)


with _𝑎_, _𝑑_ denoting the item’s discrimination and difficulty parameters, respectively, and _𝜃_ denoting the student’s skill parameter. Our
primary focus is on selecting exercises that have a probability of
a correct answer close to 0.5. This probability is determined using
pre-test results and pre-computed item parameters. We anticipate
maximal learning gains when students are presented with items
that are neither too easy nor too difficult [31]. Although we utilize this specific recommendation system in our approach, other
systems could also be integrated into this framework.
Beginning by pre-determining the item parameters _𝑎,𝑑_ for every
item related to the knowledge concept through IRT modeling (Eq. 1),
we compute the user’s skill parameter _𝜃_ based on the user’s pre-test
results. Then with the user’s skill parameter available, we select
the first three exercises whose difficulty parameter _𝑑_ is closest to _𝜃_ .
The exercises we select, along with their correct answers and
detailed explanations, are incorporated into the prompt. This allows
the LLM tutor to effortlessly refer to them and collaboratively work
through them with the student. At its core, our adaptive exercise
selection is rooted in the student’s cognitive state, as gauged by
IRT, directing our choice of exercises for each tutoring topic.


_3.2.2_ _Prompt Design for Personalized Tutoring._ In our quest to enhance the efficacy of personalization through LLM prompt-based
tutoring, we carefully design the prompt structure. Figure 2 visualizes our approach: a cyclical framework intertwining student



assessment and system prompt. The system prompt splits into two
main facets: the base prompt and personalized prompt. On the left
side of Figure 2, the student modeling component informs the system prompt, supplying inputs for the personalized prompt. This
refined system prompt subsequently guides the tutoring session.
After the session, the process transitions to the summary, wherein
a summary prompt is integrated to evaluate student interactions
throughout the session. Feedback from this summary cycles back,
updating the student assessment and reinforcing the continuous
loop of our system.
Our system employs a dual-prompt approach, comprising the
base and personalized prompts. The base prompt provides the foundational guidelines that structure the tutoring interaction. It ensures
that the tutoring session is interactive, student-centered, and adheres to effective pedagogical practices. An example of the base
prompt is highlighted in blue in Figure 2.
On the other hand, the personalized prompt varies among students, adapting to each student’s unique characteristics. It evolves
based on individual student modeling, with its initial form shaped
by the student states constructed immediately after the pre-test.
The examples in Figure 2, highlighted in orange, demonstrate how
prompts can be tailored to a student’s cognitive state, affective state,
and learning style. The personalized prompt is designed to integrate
information gathered from the pre-test and onboarding survey. For
subsequent sessions, it also includes the information of the current
student’s state based on the summaries of prior sessions to ensure
the instructions provided are relevant to the student’s progress. The
summary prompt serves as a bridge between student interaction
and GPT-4’s assessment capabilities, distilling key insights from
tutoring sessions. It is designed to extract session recaps, assess student proficiency, and suggest action items for subsequent tutoring.
The assessment and suggested actions, derived from the summary,
contribute to ongoing refinement of student modeling and the system prompt. This iterative process ensures the incorporation of
action item recommendations for tutoring strategies in the system


Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA


prompt from the second session onward. The detailed summary
prompt is available in Appendix B.2.
To offer a tangible insight into the application of our system, we
have provided an example of the entire system prompt - used with
one of the participants - in the Appendix B.1. Therein, the specifics
of the system prompt and the method of integrating summaries are
elucidated.


**4** **RESULTS**



In this section, we present several observations from the examination of our proof-of-concept tutoring system with illustrative
examples. We conducted a recruitment of 20 individuals, who exhibited a diverse range of English proficiency levels, spanning from
novices to advanced proficiency. They engaged in the complete
cycle of our tutoring system, and we conducted an analysis of the
dialogues as a basis for our findings. Table 1 provides an encompassing overview of our tutoring dataset.

|DatasetSummary|Counts|
|---|---|
|Dialogues<br>Turns<br>Avg. Turns per dialogue<br>Avg. Words per utterance (Tutor)<br>Avg. Words per utterance (Student)|74<br>1470<br>19.87<br>72.45<br>3.9|



**Table 1: Tutoring dataset summary**


We manually labeled each tutor’s utterance to understand the
dialogues’ content and nature. We categorized the utterances into
10 actions, including Greetings, Engagement, and Summary, based
on a prior study [29], with slight modifications to fit our context.
The distribution of labeled tutor actions is shown in Figure 3(a).


**4.1** **Observation**


From Figure 3(b), we examined the varying proportions of tutor
actions within a session across different students. The variation
in the appearance rate of tutor actions such as Engagement, Scaffolding, Asking for explanation, and Summary within the students
indicates that the tutoring system is successfully choosing teaching
approaches that are well-suited to each individual student.
The examples presented in tutoring dialogues further illustrate
the occurrence of personalization within the context of tutoring
sessions. These examples can be traced back to three foundational
dimensions of student assessment: cognitive state, affective state,
and learning style.
For our first example, we turn to how the system leverages
student metacognition to offer personalized teaching techniques.
In one particular scenario, a student’s self-assessed proficiency was
notably lower than what the student’s pre-test results indicated.
Recognizing this disparity, the tutoring system provides positive
reinforcement by stating:



(a)


(b)


**Figure 3: (a) Counts of labeled tutor actions across entire**
**dialogues. (b) Distribution of the ratio of each tutor action**
**computed on a student-wise basis.**


“ _It’s interesting to note that based on your pre-test re-_
_sults, I see that your knowledge about punctuation is_
_actually stronger than you think! That’s a great start._ ”

Secondly, in a situation where the student has consistently shown
high engagement, the GPT-4 has observed and suggested future
action items as,


“ _Encouraging the student to go beyond the right answer_
_and explain the reasoning behind their choices in their_
_own words proved to be an effective strategy. For fu-_
_ture sessions, continuous prompts to express thought_
_processes or reasoning behind answers could further_
_stimulate engagement and verify understanding._ ”

During the subsequent session, the tutor reflected its action items
by saying,


“ _That’s correct! The correct answer is indeed A, “NO_
_CHANGE”. You realized that “its” was the right pro-_
_noun in this scenario. Could you elaborate a bit on_
_what helped you understand that “its” should not be_
_changed?_ ”


CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim



Lastly, personalized tutoring strategies were implemented according to each student’s individual learning style. This approach
was applied not only during the initial session, where tutoring
strategies were explicitly tailored based on the reported learning
style obtained from the onboarding survey, but also in subsequent
sessions as the learning style was updated by the GPT-4. For instance, GPT-4 analyzed the student’s learning style and recommended strategies as


“ _To cater to the student’s global learning style, the tutor_
_could use summarizations and reinforcements of lesson_
_points throughout future sessions to highlight the overall_
_context and relevance of the studied matter._ ”


after the first session. During the following session, the tutor asked
the student


“ _To deepen your understanding, could you summarize_
_what you’ve just learned from this question?_ ”,


applying the recommended strategy.
Overall, the system successfully adjusted instructional strategies
in a personalized manner. Additionally, the GPT-4 demonstrated its
effectiveness in analyzing and evaluating the student states based
on the dialogues when necessary.


**4.2** **Discussion**


While our system demonstrates the potential of LLMs in conversationbased tutoring, we also uncovered limitations. A primary concern is
that precise student assessment doesn’t always result in discernible,
actionable tutoring strategies, possibly due to our system’s focus on
a question-answer format. The connection between student assessment and tailored tutoring strategies warrants closer examination.
Furthermore, maintaining student engagement in a chat interface
is challenging. This was evident from the word count discrepancy
between students and the tutor, and also from the students’ insincere responses. Addressing these concerns may require a shift from
simple dialogue generation to more engaging, interactive content.
For future work, we intend to undertake a comprehensive investigation of our system, coupled with iterative refinements based
on the outcomes obtained. Following the confirmation of personalization within our system, our subsequent focus will entail an
in-depth examination of the consequential effects, for instance in
terms of learning gain and engagement. Despite efforts to measure
learning gain using our CDM model (detailed in Appendix C), our
study revealed that the implemented tutoring system did not ensure significant learning gains. To address limitations such as short
duration and misalignment of post-test questions with tutoring
content, we plan to enhance our evaluation framework and further
investigate how our personalization leads to educational advantages. Based on this, we also plan to discern effective pedagogical
methodologies within our system, to improve our personalization
strategies from an educational perspective.


**5** **CONCLUSION**


In this paper, we present the design of a personalized tutoring system that focuses on the diagnostic components of student modeling,
leveraging the effectiveness of LLM. We elaborated on the design
considerations and described a proof-of-concept tutoring system



that was developed based on these considerations. Through this process, we have identified key areas for improvement, particularly in
connecting precise student assessments to effective tutoring strategies and enhancing user engagement. This identification of areas
for improvement sets the stage for our future research, where we
aim to refine and advance the capabilities of personalized tutoring
systems.


**REFERENCES**


[1] L. Assis, A. C. Rodrigues, A. Vivas, C. G. Pitangui, C. M. Silva, and F. A. Dorça.
2022. Relationship Between Learning Styles and Learning Objects: A Systematic
Literature Review. _International Journal of Distance Education Technologies (IJDET)_
[20, 1 (2022), 1–18. https://doi.org/10.4018/IJDET.296698](https://doi.org/10.4018/IJDET.296698)

[2] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan,
Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda
Askell, et al. 2020. Language models are few-shot learners. _Advances in neural_
_information processing systems_ 33 (2020), 1877–1901.

[3] Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric
Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al. 2023.
Sparks of artificial general intelligence: Early experiments with gpt-4. _arXiv_
_preprint arXiv:2303.12712_ (2023).

[4] Eason Chen, Ray Huang, Han-Shin Chen, Yuen-Hsien Tseng, and Liang-Yi Li.
2023. GPTutor: A ChatGPT-Powered Programming Tool for Code Explanation.
In _Artificial Intelligence in Education. Posters and Late Breaking Results, Workshops_
_and Tutorials, Industry and Innovation Tracks, Practitioners, Doctoral Consortium_
_and Blue Sky_ . Springer Nature Switzerland, Cham, 321–327.

[5] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav
Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways.
_arXiv preprint arXiv:2204.02311_ (2022).

[6] Rebecca J. Compton. 2000. Ability to disengage attention predicts negative affect.
_Cognition and Emotion_ 14, 3 (2000), 401–415.

[7] Jimmy De La Torre. 2009. DINA model and parameter estimation: A didactic.
_Journal of educational and behavioral statistics_ 34, 1 (2009), 115–130.

[8] Lim Ean Heng, Wong Pei Voon, Norazira A. Jalil, Chan Lee Kwun, Tey Chee Chieh,
and Nor Fatiha Subri. 2021. Personalization of Learning Content in Learning
Management System. In _Proceedings of the 2021 10th International Conference_
_on Software and Computer Applications_ (Kuala Lumpur, Malaysia) _(ICSCA ’21)_ .
Association for Computing Machinery, New York, NY, USA, 219–223.

[9] Moushir M El-Bishouty, Ahmed Aldraiweesh, Uthman Alturki, Richard Tortorella,
Junfeng Yang, Ting-Wen Chang, Sabine Graf, et al. 2019. Use of Felder and
Silverman learning style model for online course design. _Educational Technology_
_Research and Development_ 67, 1 (2019), 161–177.

[10] Susan E Embretson and Steven P Reise. 2013. _Item response theory_ . Psychology
Press.

[11] Richard Felder. 1988. Learning and Teaching Styles in Engineering Education.
_Journal of Engineering Education -Washington-_ 78 (01 1988), 674–681.

[12] Richard M Felder. 2002. Learning and teaching styles in engineering education.
(2002).

[13] A.C. Graesser, P. Chipman, B.C. Haynes, and A. Olney. 2005. AutoTutor: an
intelligent tutoring system with mixed-initiative dialogue. _IEEE Transactions on_
_Education_ 48, 4 (2005), 612–618.

[14] Jieun Han, Haneul Yoo, Yoonsu Kim, Junho Myung, Minsun Kim, Hyunseung
Lim, Juho Kim, Tak Yeon Lee, Hwajung Hong, So-Yeon Ahn, and Alice Oh.
2023. RECIPE: How to Integrate ChatGPT into EFL Writing Education _(L@S ’23)_ .
Association for Computing Machinery, New York, NY, USA, 416–420.

[15] Eugenia Y. Huang, Sheng Wei Lin, and Travis K. Huang. 2012. What type of
learning style leads to online participation in the mixed-mode e-learning environment? A study of software usage instruction. _Comput. Educ._ 58, 1 (2012),
338–349.

[16] Jin-Xia Huang, Yohan Lee, and Oh-Woog Kwon. 2023. DIRECT: Toward DialogueBased Reading Comprehension Tutoring. _IEEE Access_ 11 (2023), 8978–8987.

[17] David WL Hung and Der-Thanq Chen. 2001. Situated cognition, Vygotskian
thought and learning from the communities of practice perspective: Implications
for the design of web-based e-learning. _Educational Media International_ 38, 1
(2001), 3–12.

[18] Enkelejda Kasneci, Kathrin Sessler, Stefan Küchemann, Maria Bannert, Daryna
Dementieva, Frank Fischer, Urs Gasser, Georg Groh, Stephan Günnemann, Eyke
Hüllermeier, Stephan Krusche, Gitta Kutyniok, Tilman Michaeli, Claudia Nerdel,
Jürgen Pfeffer, Oleksandra Poquet, Michael Sailer, Albrecht Schmidt, Tina Seidel,
Matthias Stadler, Jochen Weller, Jochen Kuhn, and Gjergji Kasneci. 2023. ChatGPT
for good? On opportunities and challenges of large language models for education.
_Learning and Individual Differences_ 103 (2023), 102274.


Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA




[19] Bruno Leutwyler. 2009. Metacognitive learning strategies: differential development patterns in high school. _Metacognition and Learning_ 4, 2 (1 8 2009),
111–123.

[20] Jakub Macina, Nico Daheim, Sankalan Pal Chowdhury, Tanmay Sinha, Manu
Kapur, Iryna Gurevych, and Mrinmaya Sachan. 2023. MathDial: A Dialogue
Tutoring Dataset with Rich Pedagogical Properties Grounded in Math Reasoning
[Problems. arXiv:2305.14536 [cs.CL]](https://arxiv.org/abs/2305.14536)

[21] Jakub Macina, Nico Daheim, Lingzhi Wang, Tanmay Sinha, Manu Kapur, Iryna
Gurevych, and Mrinmaya Sachan. 2023. Opportunities and Challenges in Neural
Dialog Tutoring. In _Proceedings of the 17th Conference of the European Chapter_
_of the Association for Computational Linguistics_ . Association for Computational
Linguistics, Dubrovnik, Croatia, 2357–2372.

[22] Roger Nkambou, Jacqueline Bourdeau, and Riichiro Mizoguchi (Eds.). 2010. _Ad-_
_vances in Intelligent Tutoring Systems_ . Springer Berlin, Heidelberg.

[23] Benjamin D. Nye, Arthur C. Graesser, and Xiangen Hu. 2014. AutoTutor and
Family: A Review of 17 Years of Natural Language Tutoring. _International Journal_
_of Artificial Intelligence in Education_ [24 (2014), 427–469. https://doi.org/10.1007/](https://doi.org/10.1007/s40593-014-0029-5)
[s40593-014-0029-5](https://doi.org/10.1007/s40593-014-0029-5)

[[24] OpenAI. 2023. GPT-4 Technical Report. arXiv:2303.08774 [cs.CL]](https://arxiv.org/abs/2303.08774)

[25] Reinhard Pekrun. 1992. The Impact of Emotions on Learning and Achievement:
Towards a Theory of Cognitive/Motivational Mediators. _Applied Psychology_ 41,
4 (1992), 359–376.

[26] Paul R. Pintrich. 2002. The Role of Metacognitive Knowledge in Learning, Teaching, and Assessing. _Theory Into Practice_ 41, 4 (2002), 219–225.

[27] Kun Qian, Ryan Shea, Yu Li, Luke Kutszik Fryer, and Zhou Yu. 2023. User
Adaptive Language Learning Chatbots with a Curriculum. In _Artificial Intelligence_
_in Education. Posters and Late Breaking Results, Workshops and Tutorials, Industry_
_and Innovation Tracks, Practitioners, Doctoral Consortium and Blue Sky_, Ning
Wang, Genaro Rebolledo-Mendez, Vania Dimitrova, Noboru Matsuda, and Olga C.
Santos (Eds.). Springer Nature Switzerland, Cham, 308–313.

[28] Miriam Spering, Dietrich Wagener, and Joachim Funke. 2005. BRIEF REPORT.
_Cognition and Emotion_ 19, 8 (2005), 1252–1261.

[29] Katherine Stasaski, Kimberly Kao, and Marti A. Hearst. 2020. CIMA: A Large Open
Access Dialogue Dataset for Tutoring. In _Proceedings of the Fifteenth Workshop_
_on Innovative Use of NLP for Building Educational Applications_ . Association for
Computational Linguistics, Seattle, WA, USA → [Online, 52–64. https://doi.org/](https://doi.org/10.18653/v1/2020.bea-1.5)
[10.18653/v1/2020.bea-1.5](https://doi.org/10.18653/v1/2020.bea-1.5)

[30] Marcel V.J. Veenman, Pascal Wilhelm, and Jos J. Beishuizen. 2004. The relation
between intellectual and metacognitive skills from a developmental perspective.
_Learning and Instruction_ 14, 1 (2004), 89–109.

[31] Lev Semenovich Vygotsky and Michael Cole. 1978. _Mind in society: Development_
_of higher psychological processes_ . Harvard university press.

[32] Thiemo Wambsganss, Tobias Kueng, Matthias Soellner, and Jan Marco Leimeister.
2021. ArgueTutor: An Adaptive Dialog-Based Learning System for Argumentation Skills. In _Proceedings of the 2021 CHI Conference on Human Factors in_
_Computing Systems_ (Yokohama, Japan) _(CHI ’21)_ . Association for Computing
Machinery, New York, NY, USA, Article 683.

[33] Allan Wigfield and Jacquelynne S. Eccles. 2000. Expectancy–Value Theory of
Achievement Motivation. _Contemporary Educational Psychology_ 25, 1 (2000),
68–81.

[34] Kenneth Wong, Kat Leung, Reggie Kwan, and Philip Tsang. 2010. E-learning:
developing a simple web-based intelligent tutoring system using cognitive diagnostic assessment and adaptive testing technology. In _International Conference_
_on Hybrid Learning_ . Springer, 23–34.



**A** **USER FLOW OF OUR SYSTEM**


We provide a detailed description of the user flow within our system.
A user entering our system proceeds through the following stages
of the learning journey.


(1) **Onboarding** : Users start with an onboarding survey, shaping the initial student profile in our system. At the same time,
this process introduces our system’s functionalities and sets
clear expectations. The survey encompasses questions about
the student’s preferred learning style. It also asks their confidence in the subject—particularly in the SAT writing test
and concepts of Pronouns, Punctuation, and Transitions. Additionally, we collect basic demographic details for further
analysis.
(2) **Pre-test** : Upon completing the onboarding process, students
are presented with a pre-test designed to gauge their initial
understanding. This test comprises 15 questions, divided into
three knowledge concept parts of five questions each. The
test evaluates proficiency in three SAT Writing principles

   - Pronouns, Punctuation, and Transitions. Responses from
the pre-test are fed into the IRT model, enabling adaptive
exercise selection for the upcoming tutoring session. Additionally, the determined proficiency updates the student’s
affective state. After the test, students can assess their own
proficiency and preview the planned tutoring exercises.
(3) **Tutoring Session** : Personalized tutoring is delivered based
on pre-test results. Exercises for this phase are adaptively
selected based on pre-test result. The user interface splits
between learning materials and a chat with our AI tutor.
The student is encouraged to immerse themselves in this
chat-centric tutoring experience, engaging at one’s own pace
until all exercises are thoroughly covered.
(4) **Post-test** : After the tutoring session, students take a posttest on the same knowledge concept they were tutored typically consisting of five questions. Upon completion, the
system offers a side-by-side comparison, highlighting the
difference between the pre-test and post-test proficiency
levels, enabling the student to see their progress.
(5) **Iterative Learning Cycles** : After completing one round,
students choose from the remaining knowledge concepts.
They then embark on another cycle of tutoring followed by
a post-test. This iterative process continues until students
have been tutored on all three knowledge concepts.


**B** **PROMPT**

**B.1** **System Prompt**


We provide an example of the whole system prompt that was used
with one of the participants. In the initial session, we utilize the
information obtained from the onboarding survey and pre-test for
the student model. This includes proficiency levels computed from
the pre-test, as well as responses on metacognitive proficiency and
learning style gathered from the onboarding survey. In the second
and third session, the updated student model is integrated into
the system prompt, guided by the generated summary from the
preceding session.


CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim


**Session 1**


    - I want you to act as a personal English tutor.

    - Start the class by mentioning the subject of this class and asking “Are you ready to start?”

    - From now on, you’ll be teaching about Pronouns.

    - The student has self-reported his/her proficiency about this concept as “Weak”, but the measured proficiency is “Strong”.
Encourage the student by telling that he/she is better than he/she thinks according to the pre-test result.

    - Based on the Felder-Silverman Learning Style Model, the student has responded his/her learning style as “Active/Intuitive/Global”.
Considering this, apply the following teaching strategies throughout the class:
(1) Provide opportunities for students to do something active besides transcribing notes. Brief brainstorming activities might be
effective.
(2) Provide abstract concepts(principles, theories, mathematical models).
(3) Provide the big picture or goal of a lesson before presenting the steps, doing as much as possible to establish the context and
relevance of the subject matter and to relate it to the students’ experience.

    - If the student gives the correct answer, don’t immediately go to the next question. Instead, ask deeper questions regarding the
student’s learning style.

    - If the student gives the wrong answer, don’t immediately tell the right answer. Instead, ask more questions and help the student
reach to the right answer, regarding the student’s learning style.

    - Talk with the student turn by turn.

    - Ask one question in one turn.

    - Don’t explain it too long, but cut it short by asking questions.

    - Don’t ask “do you have any questions about...” but ask more meaningful questions that would encourage the student to think.

    - If the student does not understand certain concept, ask where the student is struggling. Explain the concept in an easier way so
that the student can learn the concept step-by-step.

    - The student is also provided with the same contents. Just point it with the question number, not including the whole content in
your text.

    - When the student says something irrelevant (off-topic, irrelevant to the question, etc.), request clarification.

    - If you’re done with the session, say “FINISHED.” at the very end of your last response.

    - Learning materials are provided below. Explain it based on the provided “Answer” and “Explanation”.

    - Do not make up your own examples while teaching, strictly stick to the materials provided below.

    - Before moving on to the next question, you must ask if the student has more questions and if it is okay to move on.

    - When one question is over, summarize the things the student has learned through the question.

    - If the student gives the wrong answer, point out what is wrong with it or which misconception he/she has.

    - [Learning Materials]


**Session 2**


    - I want you to act as a personal English tutor.

    - It’s this student’s second class - the student has learned about Pronouns in the previous class. I want you to welcome the student.

    - Start the class by mentioning the subject of this class and saying “Are you ready to start?”

    - From now on, you’ll be teaching about Punctuation.

    - The student has self-reported his/her proficiency about this concept as “Weak”, but the measured proficiency is “Strong”.
Encourage the student by telling that he/she is better than he/she thinks according to the pre-test result.

    - The student has learned these in the previous class. Do a review only in case the student asks about it.

    - Specific topics:
The session focused on enhancing the student’s understanding of pronouns. We explored different types of pronouns, their
functions, and the concept of pronoun-antecedent agreement using questions related to the provided passages. The student
initially had some challenges with correct pronoun use but showed noticeable improvement as the session progressed.

    - Based on the summary of the previous class, the followings are the recommended action items. Consider these throughout this
class.

**Action items regarding the student’s response level** : The student exhibited a good engagement level throughout the session,
although there were moments of hesitation. It would be beneficial to continue encouraging the student to think critically and
answer confidently. The tutor might incorporate intermediate-level quiz questions that prompt the student to identify and use
pronouns correctly.


Empowering Personalized Learning through a Conversation-based Tutoring System with Student Modeling CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA


**Action items regarding the student’s learning style** : Given the student’s Active/Intuitive/Global learning style, the tutor
should continue to incorporate brainstorming activities and abstract concepts. It may also be useful to provide more opportunities
for active participation, such as having the student come up with their own sentences using specific pronouns. The tutor should
maintain the focus on knitting both specific details and broader concepts together whenever teaching new topics.

    - The student had difficulty with the previous material. As you move on to the next concept, it’s vital to adapt the teaching
approach to mitigate further misunderstandings. These are the follow-up strategies you can implement:
(1) Encourage students to share feedback about the teaching method or areas they found challenging in the previous lesson.
Adapt based on their preferences.
(2) Use varied teaching methods to ensure the student remains engaged.
(3) Ensure that the student feels supported.
(4) Utilize analogies, real-world scenarios, and visual aids for clearer explanations.
(5) Check for understanding more frequently.

    - If the student gives the correct answer, don’t immediately go to the next question. Instead, ask deeper questions regarding the
student’s learning style.

    - If the student gives the wrong answer, don’t immediately tell the right answer. Instead, ask more questions and help the student
reach to the right answer, regarding the student’s learning style.

    - Talk with the student turn by turn.

    - Ask one question in one turn.

    - Don’t explain it too long, but cut it short by asking questions.

    - Don’t ask “do you have any questions about...” but ask more meaningful questions that would encourage the student to think.

    - If the student does not understand certain concept, ask where the student is struggling. Explain the concept in an easier way so
that the student can learn the concept step-by-step.

    - The student is also provided with the same contents. Just point it with the question number, not including the whole content in
your text.

    - When the student says something irrelevant (off-topic, irrelevant to the question, etc.), request clarification.

    - If you’re done with the session, say “FINISHED.” at the very end of your last response.

    - Learning materials are provided below. Explain it based on the provided “Answer” and “Explanation”.

    - Do not make up your own examples while teaching, strictly stick to the materials provided below.

    - Before moving on to the next question, you must ask if the student has more questions and if it is okay to move on.

    - When one question is over, summarize the things the student has learned through the question.

    - If the student gives the wrong answer, point out what is wrong with it or which misconception he/she has.

    - [Learning Materials]


**B.2** **Summary Prompt**


    - From the dialogue, summarize the overall tutoring session and suggest action items in terms of the following. The suggested
action items should be applicable in chat-based tutoring.

    - Specific topics: List up the specific topics covered throughout the class and the student’s proficiency on the corresponding topics

    - Action items regarding the student’s response level: Identify how much the student was engaged to the class, and suggest action
items regarding this. The action items should be applicable regardless of the content covered in the class.

    - Action items regarding the student’s learning style: Based on the teaching strategies suggested earlier and the student’s attitude
in the class, suggest some specific action items that can be applied in the next class.

    - If the student didn’t show any specific state, say “Unknown”

    - You must reply in this form:
*Specific topics:
*Action items regarding the student’s response level:
*Action items regarding the student’s learning style:


CHI EA ’24, May 11–16, 2024, Honolulu, HI, USA Minju Park, Sojung Kim, Seunghyun Lee, Soonwoo Kwon, and Kyuseok Kim



**C** **EXPERIMENTAL RESULTS**


We present the detailed specifications pertaining to the system analysis. In our future work, it is anticipated that this comprehensive
methodology applied in the system analysis will be employed in a
more methodical experimental design which aims to explore novel
research inquiries.


_C.0.1_ _Results of Adaptive Exercise Selection._ In our implementation, we applied the IRT model to each of the three knowledge concepts, utilizing our private interaction dataset, which encompassed
Pronouns (4,669 interactions), Punctuation (12,682 interactions),
and Transitions (5,931 interactions). The dataset was utilized to
train the IRT model and establish the item parameters required for
calculating the student’s proficiency. This process resulted in an
average Area Under the Receiving Operator Curve (AUC) of 0.65.
We found that the overall correctness ratio was 0.57 for the exer
cises presented during the tutoring session. For this calculation, we
considered only the first response when assessing the correctness
of a covered question. Noting that the system is designed to select
exercises with a probability that each student will get the right
answer close to 0.5, this result indicates that the adaptive exercise
selection mechanism successfully operated as intended, aligning
the exercise difficulty with the students’ proficiency levels.



_C.0.2_ _Learning Gain._ To quantify the learning gains for each knowledge concept, we used the student’s skill parameters from IRT: _𝜃𝑝𝑟𝑒_
for the pre-test and _𝜃𝑝𝑜𝑠𝑡_ for the post-test. By applying these parameters to our pretrained set of questions within the concept, we
calculated the average correctness probabilities, resulting in _𝑝𝑝𝑟𝑒_
and _𝑝𝑝𝑜𝑠𝑡_ . The measure of learning gain is defined as the difference between average correctness probabilities: _𝑝𝑝𝑜𝑠𝑡_ − _𝑝𝑝𝑟𝑒_ . This
provides a more intuitive understanding of learning gain than the
difference in skill parameters, _𝜃𝑝𝑜𝑠𝑡_ − _𝜃𝑝𝑟𝑒_ .
For all 20 participants, the average learning gain, _𝑝𝑝𝑜𝑠𝑡_ − _𝑝𝑝𝑟𝑒_,
was calculated as −0 _._ 0753, 0 _._ 0159, and −0 _._ 0102 in Pronouns, Punctuation, and Transitions, respectively. Contrary to our expectations,
our study indicated that the implemented tutoring system did not
guarantee notable learning gains. Several factors could contribute
to these results. Firstly, the relatively short span of tutoring sessions may not be sufficient to yield significant learning gain. Secondly, measuring the impact of tutoring might be more plausible
if post-test questions mirrored the content covered during tutoring sessions, directly checking the understanding and applicability
of taught concepts. However, when broadly assessing proficiency
within a knowledge concept, substantial learning gains might not
manifest, even with master human tutoring. Indeed, to truly discern the effectiveness of personalization and adapt our tutoring
system accordingly, a careful approach of measuring learning gains
is essential.


