## ADAPTIVE LEARNING PATH NAVIGATION BASED ON KNOWLEDGE TRACING AND REINFORCEMENT LEARNING

Jyun-Yi Chen


National Taiwan Normal University
61175017h@ntnu.edu.tw


Saeed Saeedvand


National Taiwan Normal University
saeedvand@ntnu.edu.tw


I-Wei Lai


National Taiwan Normal University
iweilai@ntnu.edu.tw


ABSTRACT


This paper introduces the Adaptive Learning Path Navigation (ALPN) system, a novel approach for
enhancing E-learning platforms by providing highly adaptive learning paths for students. The ALPN
system integrates the Attentive Knowledge Tracing (AKT) model, which assesses students’ knowledge states, with the proposed Entropy-enhanced Proximal Policy Optimization (EPPO) algorithm.
This new algorithm optimizes the recommendation of learning materials. By harmonizing these
models, the ALPN system tailors the learning path to students’ needs, significantly increasing learning effectiveness. Experimental results demonstrate that the ALPN system outperforms previous
research by 8.2% in maximizing learning outcomes and provides a 10.5% higher diversity in generating learning paths. The proposed system marks a significant advancement in adaptive E-learning,
potentially transforming the educational landscape in the digital era.


Keywords E-learning · Adaptive Learning · Knowledge Tracing · Deep Learning · Deep Reinforcement Learning


1 Introduction


Integrating pedagogy and technology has brought significant transformations in education, with E-learning systems
emerging as accessible, cost-effective, collaborative, and flexible alternatives to classroom-based education [1, 2, 3, 4].
E-learning systems enable students to access a wealth of learning materials for self-directed learning at their convenience. Considering these advantages, exploring strategies to enhance E-learning systems further and optimizing their
effectiveness becomes crucial. One promising approach involves offering students well-designed learning paths on
these systems. With the support of learning paths, students have a lower chance of encountering learning disorientation and cognitive overload, which improves their learning efficiency [5, 6, 7, 8].


A learning path constitutes an automatically curated sequence of learning materials to ensure the student has all the
required knowledge to achieve their learning goal [9]. Depending on the student’s requirements, learning materials
can be courses, topics, or learning objects [10]. The methods of selecting and organizing learning materials to form a
learning path have been studied for some time. Traditional E-learning systems commonly utilize standardized methods


PRIME AI paper


to provide fixed learning paths [11, 12, 13, 14, 15, 16]. However, these methods may make some students feel underchallenged or overwhelmed, as each student has a different learning goal and knowledge background [17]. Hence,
providing adaptive learning paths tailored to individuals has become a pressing concern.


In recent years, researchers have proposed different approaches for adaptive learning path planning using various
personalization parameters. These parameters include learning goals [18, 19], time limitations [20, 21], knowledge
backgrounds [22, 23], etc. The learning goals can be deadline-driven or mastery-driven, determined by if the students
have time limitations. The knowledge backgrounds refer to the knowledge level of students before they engage with
the learning paths. When an adaptive learning path planning approach can accurately assess students’ knowledge
backgrounds in advance, it can recommend appropriate learning materials based on their deficiencies. However, as
the number of students and learning materials on E-learning systems continues to grow, maintaining or improving the
system’s ability to diagnose students’ learning progress becomes challenging [24]. To address this problem, developing
scalable methods stands as a permanent solution.


This paper proposes an Adaptive Learning Path Navigation (ALPN) system that recommends learning materials to
students according to their knowledge states, i.e., the mastery level of concepts in a subject. The system employs a
Knowledge Tracing (KT) model to quantify students’ current knowledge states [25], followed by a decision-making
model that recommends tailored learning materials. As students complete the learning materials, the KT model updates their knowledge states by assessing their responses. This iterative recommendation and knowledge state updating
process continues until students achieve their learning goals. Throughout this process, the learning materials recommended by the decision-making model form the students’ learning paths. By considering the dynamics of a student’s
knowledge state during the learning process, the proposed ALPN system generates highly adaptive learning paths. In
addition, we implement two models using deep learning techniques to enhance scalability, enabling our system to
maintain or improve its performance when facing more diverse students and learning materials [26, 27].


KT estimates students’ knowledge states and predicts their future performance according to their learning records
within an E-learning system [28]. Utilizing deep learning techniques, KT has achieved significant advancements
in learning diagnosis [29]. Our system employs an attention-based model, Attentive Knowledge Tracing (AKT), to
offer reliable assessments of students’ learning progress [30, 31]. For another part, we propose an Entropy-enhanced
version of Proximal Policy Optimization (PPO) called EPPO for learning material recommendations [32]. EPPO
demonstrates superior performance in optimizing the student’s learning outcome compared to vanilla PPO in our task
by incorporating enhanced exploratory capabilities. With the combination of AKT and EPPO, our ALPN system can
create effective learning paths.


In the experiments, we compared our proposed ALPN system with the Knowledge Tracing based Knowledge Demand Model (KT-KDM), currently the method most similar to ours in learning path recommendation research [33].
Regarding maximizing students’ learning outcomes, our ALPN system outperformed KT-KDM on average by 8.2%.
Moreover, the ALPN system exhibited a 10.5% higher diversity in generating learning paths than KT-KDM. We also
conducted additional analyses to demonstrate the performance of the ALPN system in various aspects.


In a nutshell, the proposed ALPN system integrates AKT and EPPO models, offering adaptive learning paths to
students. The contributions of this work are listed below:


1. Proposing the EPPO, an algorithm outperforming conventional PPO in learning path recommendation.


2. Developing a system that can improve students’ learning outcomes by 8.2% more than the previous similar
approach.


3. Applying a KT model based on the attention mechanism to assess a student’s knowledge state thus makes the
generated adaptive learning paths more reliable.


4. Creating a framework that can continuously update a student’s knowledge state, enabling the system to consistently recommend the most suitable learning materials based on their needs.


The paper presents an organization: Section 2 overviews the relevant existing works, Section 3 describes the mechanism of KT, and Section 4 elaborates on the learning path planning method. Next, Section 5 introduces the dataset and
presents the experimental results. Finally, Section 6 concludes this paper and proposes directions for future research.


2 Related Work


This section reviews relevant prior work in knowledge tracing and reinforcement learning.


2


PRIME AI paper


2.1 Knowledge Tracing


KT models have two categories: traditional knowledge tracing models and deep learning-based knowledge tracing
models.


    - Item Response Theory: Item Response Theory (IRT) is a popular probabilistic framework in psychometrics
that assesses latent traits based on individuals’ responses to a collection of items [34]. Although IRT demonstrates robustness across diverse domains, it falls short in capturing the dynamic nature of learning due to the
assumption that the latent trait (e.g., a student’s knowledge level) remains constant over time. Furthermore,
IRT does not explicitly model the sequence of item interactions, a critical aspect for comprehending learning

processes.


   - Bayesian Knowledge Tracing: Bayesian Knowledge Tracing (BKT) addresses the limitations of IRT by representing a student’s learning process as a hidden Markov model [28]. This modeling technique monitors
an individual’s proficiency in knowledge components over time, considering the sequential order of item interactions. Nonetheless, BKT permits only a binary representation of knowledge mastery (i.e., learned or
unlearned), indicating that its effectiveness warrants further enhancement.


   - Deep Knowledge Tracing: Deep Knowledge Tracing (DKT) expands the application of deep learning in
knowledge tracing, employing Recurrent Neural Networks (RNN) or Long Short-Term Memory (LSTM)
to model individuals’ dynamic knowledge acquisition [29]. By leveraging neural network computations,
DKT can represent a more extensive range of knowledge states. Consequently, it offers a nuanced and
precise approximation of a student’s learning progress, demonstrating the potential for adaptive learning path
recommendations.


2.2 Reinforcement Learning


In recent years, Reinforcement Learning (RL) has emerged as a prominent field within artificial intelligence and
machine learning. It aims to develop algorithms that allow agents to learn optimal policies through interactions with
their environment, aiming to maximize cumulative rewards [35]. Classic RL algorithms, such as Q-learning and
SARSA, rely on tabular representations of state-action values, which may suffer from the curse of dimensionality
in cases with large state spaces [36, 37]. This limitation has led to exploring function approximation techniques to
generalize across states and actions, such as linear function approximation and tile coding [38, 39]. However, these
methods often fail to scale well or learn efficiently in complex environments with high-dimensional states and action

spaces.


Deep Reinforcement Learning (DRL) was introduced to overcome traditional RL methods’ limitations, integrating
deep learning techniques with reinforcement learning algorithms. DRL leverages the power of deep neural networks
to represent the value functions or policies, enabling the learning of complex features and representations in highdimensional spaces [40]. One breakthrough example is the Deep Q-Network (DQN) algorithm, which successfully
learned to play a wide range of Atari games directly from raw pixel inputs, outperforming many previous methods [41].
DRL has also demonstrated remarkable success in domains such as robotic control [42], natural language processing

[43], and game-playing [44]. Unlike traditional RL methods, DRL offers significant improvements in scalability and
generalization, enabling agents to learn more effectively in complex, high-dimensional environments.


3 Knowledge Tracing Model


To ensure that each learning material provided by the system is most effective for the students, we need to track their
learning progress as they follow the learning path. In this regard, we employ the KT technique to assess the student’s
knowledge state continuously. We describe the central concepts of KT in this section and briefly introduce the KT
approach we have adopted.


3.1 Problem Definition


KT quantifies the likelihood that a student will correctly answer specific exercises based on the student’s learning
history. Given a set of exercise indices E = {e1, e2, . . ., eJ } ∈ N and a student’s historical interaction sequence
I = (I1, I2, . . ., It) ordered by time. For each interaction log It = (et, ct), et ∈E denotes the exercise index that the
student answered at time t, and ct ∈{0, 1} represents the correctness of the answer. Using these interaction logs as
input, the KT model outputs a vector st = [s1,t, s2,t, ..., sJ,t] [⊤] that represents the student’s knowledge state at time t.
Each element in the vector st represents the probability of the student correctly answering a specific exercise in E:


3


PRIME AI paper


Figure 1: The workflow of the proposed ALPN system.


sj,t = P[cj = 1|I1, I2, ..., It−1, ej]. (1)



We consider the knowledge state to represent the student’s knowledge background and utilize it as a personalization
parameter to influence the recommendation of the subsequent learning path.


3.2 Applied Scheme


In this study, we are motivated by the AKT scheme [31] to employ the KT model for assessing students’ knowledge
state. The AKT employs an exercise self-attentive encoder to convert each input exercise index into a contextualized
representation. Then, it evaluates the student’s acquired knowledge from past interactions by using a knowledge
acquisition self-attentive encoder. Following the knowledge acquisition step, the AKT filters the previously acquired
knowledge to extract the relevant knowledge associated with the current exercise utilizing a single attention-based
knowledge retriever. Finally, it evaluates the probability of the student correctly answering a specific exercise using
the extracted knowledge.


4 Learning Path Navigation System


The proposed ALPN system employs a pre-trained AKT model as the environment, allowing the decision-making
model, i.e., the agent, to explore diverse learning paths. When the student follows a learning path, multiple stateaction-reward tuples will be generated over time, collectively forming a trajectory:


τ = {(s1, a1, r1), (s2, a2, r2), . . ., (sT, aT, rT )}, (2)


where st denotes the student’s knowledge state, at represents the exercise index recommended by the agent, and rt
signifies the immediate reward that the agent received after the student completes the exercise. We assign the maximum
length of a single learning path to Tmax, such that T ≤ Tmax. Notably, s1 indicates the knowledge background the AKT
evaluates before the student engages with the learning path.


In Figure 1, we present the conceptual view of our ALPN system framework. First, AKT analyzes the student’s
learning history and determines their initial knowledge state. Subsequently, the agent takes this knowledge state as
input and recommends appropriate learning material (exercise) to the student. Upon completion of the exercise by the
student, the system incorporates the interaction record into their learning history and reevaluates the knowledge state.
This process continues until the student’s knowledge level surpasses the predetermined learning goal. The following
subsections will delve into the various components employed in the system’s implementation.


4.1 Probabilistic Formulation


Our system’s primary objective is to provide learning paths to students with various knowledge backgrounds, ensuring
they achieve their learning goals. The optimal approach involves augmenting the probability of generating trajectories
with the highest cumulative reward across heterogeneous initial knowledge states. Hence, we commence by defining
the probability formulation:



Pθ[τ ] = P[s1]



T
� πθ(at|st)P[ct|st, at]P[st+1|st, at, ct], (3)


t=1


4


PRIME AI paper


where P[s1] indicates the initial knowledge state distribution, and P[st+1|st, at, ct] stands for the transition probability
of knowledge state. The term P[ct|st, at] represents the Bernoulli distribution of the probability that the student
correctly answers the exercise at, as determined by the AKT model.


4.2 Learning Goal


In our ALPN system, we set the primary learning goal for the student as maximizing their learning level. We define
the student’s learning level as the average pass rate (APR) for all exercises. We can express the student’s APR at time
t as:



1
APRt = |A|



|A|
� sj,t, ∀j : sj,t ∈ (0, 1), (4)

j=1



where |·| denotes the cardinality of a set, which represents the number of elements in the set. Specifically, |A| represents the number of available exercises in the E-learning system and denotes the action space size for the agent. sj,t
indicates the probability of a student answering the j-th exercise in the exercise set correctly at time t.


Upon completion of an exercise that the agent recommends, the student’s APRt undergoes comparison with a predetermined threshold β. When the APRt exceeds or equals β, the system deems the student to have accomplished the
learning goal. The β does as a hyperparameter, allowing for adjustments as required.


4.3 Reward Function


We develop the reward function to align with the student’s learning goal. It can evaluate the quality of the ALPN
agent’s actions and guides its decision-making by reinforcing actions that yield desirable results. We emphasize the
learning gain as a critical parameter to achieve this. Learning gain refers to the distinction between students’ skills,
competencies, content knowledge, and personal development at two different points in time [45]. It varies from
learning outcomes, as learning gain compares performance at two-time points, whereas learning outcomes focus on
knowledge level after finishing a learning session. Precisely, we calculate learning gain as LGt = APRt − APRt−1,
where APRt and APRt−1 indicate a student’s current knowledge level and previous knowledge level.


This study also considers the distance between a student’s current knowledge level and the learning goal, as an essential
parameter. We define the distance parameter as dt = β − APRt, which acts as a factor influencing the reward signal’s
magnitude. As the distance parameter gets incorporated, dividing the learning gain by distance returns the central part
of the reward signal, reflecting the challenge associated with further enhancing the student’s knowledge level when
approaching their learning goal.


Furthermore, we regard the diversity in learning materials (i.e., exercises) as essential within a single learning path.
We apply a penalty parameter λ into the reward function to ensure diversity, with the definition as:


λ = [d][1][ |A|], d1 = β − APR1, (5)

Tmax


where d1 denotes the initial distance between the student’s knowledge level and the learning goal. nj,t indicates how
many times the agent has recommended the j-th exercise in the current learning path at time t.


With these parameters, we formulate the reward function as follows:



LGt |A|



(6)

t |A|

dt if LGt < 0.



dt − λ [n][j,t] if LGt ≥ 0



rt =








LGt |A|



Deserving of mentioning exists that a more elevated initial knowledge level of the student (i.e., a smaller d1) corresponds to a less penalty parameter value. Reducing constraints on the agent allows for increased exploration opportunities, assessing the potential benefits of student review for specific exercises.


5


PRIME AI paper


4.4 Maximize Expected Reward


The ALPN agent learns a target policy πθ(·|st) that correlates the student’s knowledge state st ∈S to a distribution
over all actions a ∈A, with the objective of maximizing the expected cumulative rewards:


R¯θ = Eτ ∼πθ [R(τ )] = � R(τ )Pθ[τ ]. (7)


τ


In order to achieve maximization, we require the computation of its gradient:



∇R [¯] θ = �



R(τ )∇Pθ[τ ] = �

τ τ



R(τ )Pθ[τ ]∇ log Pθ[τ ]

τ



τ τ (8)

= Eτ ∼πθ [Rτ ∇ log Pθ[τ ]].



However, computing the specific expected value proves infeasible. Thus, we approximate it by sampling a sufficient
number N of trajectories:



∇R [¯] θ ≈ [1]

N


= [1]

N



N
� R(τ [n] )∇ log Pθ[τ [n] ]


n=1



N
�


n=1



T
� R(τ [n] )∇ log(πθ(a [n] t [|][s][n] t [)][P][[][c][n] t [|][s][n] t [, a][n] t [])][,]


t=1



(9)



where the probability P[st+1|st, at, ct] in (3) does not feature, as it stems from the trained AKT model in the environment. The knowledge state transition outcome exhibits determinism for a given pair (st, at, ct). Namely,
log P[st+1|st, at, ct] reduces to 0, rendering its inclusion in (9) redundant.


With the gradient, we employ the gradient ascent algorithm to optimize the parameters θ of the policy network, resulting in the identification of the optimal policy πθ [∗][(][·|][s][t][)][, which can maximize the students’ long-term learning gains.]


4.5 Advantage Function


From a practical implementation standpoint, when a student answers a recommended exercise correctly, the obtained
learning gain is predominantly positive. This circumstance leads the agent to need help identifying the most beneficial
exercises for the student in the long run, as it frequently receives reward signals greater than 0. Consequently, we
utilize the following advantage function for replacing the original R(τ ):


Aθ(st, at) = Σt [T][′] =t [(][γ][t][′][−][t][r][t][′] [)][ −] [V][θ][(][s][t][)]
(10)
= Qθ(st, at) − Vθ(st),


where γ ∈ [0, 1] denotes the discount factor, which balances the immediate rewards with the potential long-term
rewards resulting from a particular action.


Operating this advantage function allows the agent to ascertain how recommending a specific exercise proves more
effective than suggesting other exercises in a given knowledge state.


4.6 Policy Gradient Method


We adopt the Proximal Policy Optimization (PPO) algorithm [32] as the foundation to construct our ALPN agent.
PPO has been widely recognized for its exceptional reinforcement learning (RL) task performance, offering enhanced
sample efficiency and training stability advantages. In this section, we present a comprehensive explanation of the
PPO algorithm and describe its implementation within the framework of our proposed ALPN agent.


Figure 2 illustrates the network architecture of the ALPN agent. The architecture comprises two parts: the actor
and critic networks. The actor network determines the optimal policy that dictates the agent’s actions in the AKT
environment. It receives the input st and generates a probability distribution πθ(·|st) over the action space A, guiding
the agent’s decisions. The primary objective of the actor entails maximizing the expected cumulative reward R [¯] θ when
students follow their learning paths. For another part, the critic network serves as a value function estimator that


6


Actor



PRIME AI paper


πθ(·|st)



st











Figure 2: The network architecture of the ALPN agent.


calculates the value of the student’s current state. Given the state st and policy πθ, this network takes the same input
as the actor but outputs a scalar value Vθ(st) indicating the expected return. The critic aids in reducing the variance in
the policy gradient estimate, thus enhancing the stability and convergence of the agent’s learning process.


We utilize a replay buffer and the PPO’s Clipped surrogate objective function to enhance the agent’s sample efficiency.
By incorporating a replay buffer, we can store past experiences, allowing the agent to learn from these experiences
more data-efficiently. The Clipped surrogate objective function performs as a critical component in the PPO, addressing the central challenge of policy optimization [40, 46, 35]: balancing stable updates with sample efficiency.
This function strives to prevent the policy from undergoing substantial updates, which could result in instability and
suboptimal performance. We formulate the Clipped surrogate objective function as follows:


L [CLIP] θ [k][ (][θ][) =][ E][(][s] t [,a] t [)][∼][π] θk [[min(][ρ][t][(][θ][)][A][θ][k] [(][s][t][, a][t][)][,][ clip(][ρ][t][(][θ][)][,][ 1][ −] [ǫ,][ 1 +][ ǫ][)][A][θ][k] [(][s][t][, a][t][))]][,] (11)


where θ [k] means that we use the old policy πθk to interact with the environment to collect transitions and compute
advantage Aθk (st, at). Then, ρt(θ) indicates the probability ratio, which measures the relative likelihood of selecting
a particular action under the new policy compared to the old policy:



ρt(θ) = [π][θ][(][a][t][|][s][t][)] (12)

πθk (at|st) [.]



The probability ratio updates the policy, maintaining stability and ensuring consistency between the new and old
policies. The PPO algorithm employs the Clipped surrogate objective function, which utilizes the probability ratio to
balance stable policy updates and sample efficiency. Clipping the probability ratio within a specific range [1 − ǫ, 1 + ǫ]
enables to avoid substantial policy updates that could cause instability and poor performance, where ǫ denotes a
hyperparameter typically set to 0.2.


We can further discuss the incorporation of two additional components in the objective function: the value function
error term and the entropy bonus. These elements desire to address specific aspects of the learning process and enhance
the overall performance of the ALPN system.


Incorporating the value function error decreases the mistake in value estimation from the critic, reducing the variability in the advantage function estimation process [47]. This enhanced precision, in turn, helps the agent make better
decisions while navigating the learning path. Besides, when utilizing a neural network architecture that shares parameters between the actor and critic, operating an objective function that amalgamates the policy surrogate and a value
function error term becomes essential.


The entropy bonus, counted to foster exploration and avert premature convergence to a suboptimal policy, encourages
the agent to balance exploration and exploitation [48, 49]. This balance facilitates the discovery of novel strategies and
elevates the agent’s overall performance. Moreover, the entropy term helps the agent evade local optima and maintain
learning progress throughout training.


7


PRIME AI paper


Upon integrating these terms, the resulting objective function emerges, which the ALPN agent maximizes:


Lθk(θ) = E(st,at)∼πθk [L [CLIP] θ [k][ (][θ][)][ −] [1] 2 [L] θ [VF][k] [(][θ][) +][ αH][[][π][θ][k] [](][s][t][)]][,] (13)


where L [VF]
θ [k] [(][θ][) = (][V][θ][k] [(][s][t][)][ −] [V][θ][(][s][t][))][2][ and][ H][[][π][θ][k] [](][s][t][)][ represents the entropy bonus.][ α][ denotes the temperature,]
which controls the trade-off between exploration and exploitation. A higher temperature value promotes exploration
by increasing the randomness in the agent’s recommendations. In contrast, a lower temperature value encourages
exploitation by focusing more on the agent’s current understanding.


4.7 Entropy-Enhanced Proximal Policy Optimization


Typically, when we incorporate the entropy bonus as part of the objective function, it is calculated based on the current
policy. However, in this study, we explored an alternative approach. We stored the entropy computed by the policy
at each timestep and the transitions in the replay buffer. In the objective function, we included the entropy from the
buffer during policy updates. This approach allowed the agent to exhibit higher exploratory behavior in the early
stages of training, facilitating the discovery of more suitable policies. We referred to the modified PPO with this
adjustment as Entropy-enhanced Proximal Policy Optimization (EPPO). Through our experiments, we demonstrated
the effectiveness of EPPO in providing optimal learning paths and improving the diversity of learning paths.


Finally, we provide a formal algorithmic description of the entire ALPN system comprising EPPO:


Algorithm 1 Adaptive Learning Path Navigation (ALPN) with EPPO

1: Initialize agent parameters θ [1]

2: Initialize replay buffer D
3: for k = 1, 2, . . . do
4: Input a student’s learning history I
5: Input a student’s initial knowledge state s1
6: for t = 1, 2, . . ., Tmax do
7: Recommend exercise at on policy πθk (·|st)
8: Set ct based on knowledge state st
9: Update learning history I with at, ct
10: Get st+1 by I and AKT model
11: Compute reward signal rt
12: Store transition (st, at, rt, st+1) and entropy bonus H[πθk ](st) in D
13: if APRt ≥ β then
14: break

15: end if

16: end for
17: Sample transitions and entropy bonuses from D
18: Estimate advantages Aθk (s1, a1), Aθk (s2, a2), . . ., Aθk (sT, aT )
19: Compute objective function Lθk(θ)
20: Update policy parameters θ [k][+1] = arg maxθ Lθk(θ)
21: end for


5 Experiments


In this section, we will examine the empirical evaluation of our proposed ALPN system. The section begins with
introducing the dataset employed to train the AKT model within our system. Next, we will elucidate the evaluation
methodology employed to assess the system’s performance and present the agent’s training process results.


Subsequently, we will provide additional analyses, including insights into the changes in the learning path length
throughout the training process and an exploration of the diversity within the generated learning paths. These analyses
will contribute to a more profound understanding of the ALPN system’s efficacy and potential implications for adaptive
learning.


8


PRIME AI paper


5.1 Dataset


In our research, we train the AKT model within the ALPN system utilizing the Junyi Academy Math Practicing Log
(Junyi) dataset [50]. Comprising 25,925,922 interaction logs from 247,606 real-world students across 722 different
math exercises, this dataset presents an exhaustive set of learning interactions. Each dataset exercise is designated
with topic and area information, with 40 varied topics per exercise, such as absolute value, circle properties, and
fractions. Areas represent broader categories, each encompassing several topics, and seven distinct areas are included,
e.g., arithmetic, logic, and algebra. This dataset is open-sourced by Junyi Academy, one of Taiwan’s leading online
education platforms.


5.2 Learning Outcome


We evaluate the performanceof our system by concentrating on the student’s learning outcomes APRT after completing
the learning paths provided by the agent. This performance metric offers a practical insight into the agent’s capability
to generate adaptive learning paths that effectively improve students’ knowledge states.


In the following experiments, we refer to the ALPN system that utilizes EPPO as ALPN-EPPO, and compare it with
various variants of ALPN and a baseline:


   - ALPN-PPO: ALPN-PPO is the ALPN system that uses vanilla PPO for learning material recommendations.


   - ALPN-A2C: ALPN-A2C represents the ALPN system that applies the Advantage Actor-Critic (A2C) algorithm.


   - KT-KDM: KT-KDM, similar to our proposed ALPN system, employs the Deep Knowledge Tracing (DKT)
method to track students’ knowledge and utilizes A2C to provide learning paths [33].


We compared ALPN-EPPO with other systems in terms of their performance during the training process over 3000
episodes. In each episode, a student was randomly sampled as the target for learning path recommendation, and
each student had different learning records. With all students having a learning goal set to 0.8, the result is shown in
Figure 3. The figure shows that the ALPN-EPPO system is the most effective and robust in helping students achieve
their learning goals. In comparison, the ALPN systems using general PPO or A2C and KT-KDM exhibit weaker
performance. Additionally, it is worth noting that despite both KT-KDM and ALPN-A2C using the A2C algorithm
for their decision-making models to recommend learning materials, there is a significant difference in convergence
speed and stability between them. This discrepancy is attributed to the differences in the learning diagnosis models
employed by the two systems, i.e., the differences between DKT and AKT.


We conducted a statistical analysis on the initial knowledge levels of the 3000 sampled students during the training
process in ALPN-A2C and KT-KDM, and the results are depicted in Figure 4. We observed that, for the same group of


0.85


0.8


0.75


0.7



0.65


0.6


0.55


0.5





0.45

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||
|||||||||||||||
|||||||||||||||
||||||||||Lear<br>ALPN|ning goa<br>PPO|l β = 0.<br> KT E|8<br>  PO||
||||||||||-<br>ALPN|(<br>-PPO (A|+<br> KT + PP|)<br>  O)||
||||||||||ALPN-|A2C (A|KT + A2|C)||
||||||||||KT-K|DM (DK|T + A2|C)|C)|
|||||||||||||||

0 250 500 750 1,000 1,250 1,500 1,750 2,000 2,250 2,500 2,750 3,000


Episode


Figure 3: The students’ learning outcomes during the agents’ training process.


9


PRIME AI paper



1,000


800


600





400


200


0

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||KT-<br>ALP|KDM (DKT +<br>N-A2C (AKT|A2C)<br> + A2C)||||||
||||||||||
||||||||||
||||||||||
||||||||||

0 0.1 0.2 0.3 0.4 0.5 0.6 0.7


Initial knowledge level


Figure 4: The initial knowledge state distribution of students during the training process for ALPN-A2C and KT-KDM.


students, the ALPN-A2C system using AKT could assess their knowledge background more comprehensively. On the
other hand, the KT-KDM system using DKT provided similar evaluation results for the students. It means there is a
significant difference in the standard deviations of the output distributions between the DKT and AKT. In other words,
the complexity level varies when the two KT models are used as environments. Therefore, the KT-KDM system with
DKT as the environment is able to converge faster during training, as its environment exhibits a lower complexity
level. However, this also indicates that DKT struggles to estimate higher levels of knowledge, regardless of how well
students perform in their learning activities.


Thus, our ALPN system incorporates AKT, allowing for a more nuanced assessment of student’s knowledge levels.
However, we must employ more exploratory DRL algorithms to tackle a more complex environment. In this regard,
our proposed EPPO algorithm is the optimal choice.


5.3 Learning Efficiency


In addition to continuously providing effective learning paths for students with diverse knowledge backgrounds, students’ learning time is also an important aspect to consider. Therefore, we examined whether the ALPN system could
learn to recommend fewer learning materials during the training process while still helping students achieve the same
learning goals.


Figure 5 illustrates the variations in the length of learning paths generated by different systems as the number of
training episodes progresses. It represents the changes in the number of learning activities students undertake to
achieve their learning goals. From the figure, we observe that ALPN-EPPO enables students to achieve their learning
goals with the fewest number of learning activities. It indicates that our proposed EPPO algorithm has a better impact
on improving students’ learning efficiency.


60



55


50


45


40





35


30


25


20

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||
|||||||||||||||
||||||||||KT-K<br>ALPN|DM (D<br>A2C A|T + A2<br> KT + A|C)<br>  C||
||||||||||<br>ALPN|- (<br>PPO|<br> KT P|)<br>  O||
||||||||||<br>ALPN|- (<br>PPO|+<br> KT E|)<br>  PO||
||||||||||-|(|+|)|)|
|||||||||||||||
|||||||||||||||
|||||||||||||||
|||||||||||||||

0 250 500 750 1,000 1,250 1,500 1,750 2,000 2,250 2,500 2,750 3,000


Episode


Figure 5: The number of attempts students made to answer exercises in the learning path during the agent’s training

process.


10


PRIME AI paper


5.4 Diversity


In our ALPN system’s reward function, we designed a penalty parameter λ specifically to promote learning path
diversity. This parameter assists the system in learning how to recommend appropriate learning materials for various
knowledge states while maintaining diversity. A high level of diversity indicates a low repetition rate of learning paths.
It indicates that the learning materials are recommended according to various knowledge states. Here, to evaluate the
diversity in our ALPN system, we use the metric DIV [51]:


̸



̸ 1 − [P][i][ ∩] [P][j]

� li



̸

�



Σi≠ j

DIV =



̸

N − 1, (14)



̸

li



̸


where Pi and Pj represent the learning paths provided by the system to student i and student j, respectively. li denotes
the length of learning path Pi, and N represents the total number of learning paths. A higher DIV value indicates a
greater diversity in the learning paths provided by the system. We summarized the DIV for each system in Table 1,
and observed that all ALPN systems exhibited higher diversity than KT-KDM, with ALPN-EPPO being the most
prominent.


We also examined whether the ALPN system would provide different learning paths for students with similar knowledge levels. It also reflects that the system can generate learning paths with high diversity. We use the trained ALPNEPPO to sample two students with very close initial knowledge levels and offer them individual learning paths. Figure 6 display the result of the comparison between the learning paths of the two students. The figure illustrates the
changes in knowledge levels for student A and student B along their respective learning paths. The nodes on the graph
represent the learning materials, i.e., the exercises, that students received at that timestep. We use pre-defined categories from the Junyi dataset to annotate each exercise with a color corresponding to its area. Red asterisks indicate
the timesteps at which students achieve learning goals. From this figure, even when these two students have a very
similar initial knowledge level, there are still specific differences in the learning paths they receive. Moreover, both
students can successfully achieve their learning goals. It indicates that our ALPN-EPPO system helps students achieve
their goals and exhibits diversity in generating learning paths.


Table 1: Comparison among learning path planning methods.


Methods KT-KDM ALPN-A2C ALPN-PPO ALPN-EPPO


Diversity (DIV) 0.881 0.952 0.965 0.968


algebra arithmetic geometry analytic-geometry probability-statistics calculus end



̸


0.9


0.8


0.7


0.6


0.5


0.4


0.3


0.2



̸

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|Col28|Col29|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
|||||||||||||||||||||||||St|d|t A|||
|||||||||||||||||||||||||St|ue<br>de|<br>t B|||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||
||||||||||||||||||||||||||||||



0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26


Timestep (t)


Figure 6: The students’ learning paths with similar initial knowledge levels.


11


PRIME AI paper


5.5 Cumulative Reward


We can utilize the cumulative reward obtained during training by the system as another important evaluation metric
because it represents the system’s overall performance in an environment. Figure 7 illustrates the changes in cumulative reward during the training process for our ALPN system using EPPO, PPO, and A2C, respectively. We can
observe that ALPN-EPPO achieves the highest cumulative reward, indicating the outstanding performance of EPPO
in our learning path recommendation task. Furthermore, since our reward function differs from the one used in the
KT-KDM system, we do not compare it.


5.6 Learning Process


Next, we sampled three additional students and used the ALPN-EPPO system to provide them with learning paths.
The changes in their knowledge levels during the learning process are depicted in Figure 8. The graph shows that not
every student’s knowledge level consistently increases. For instance, student C’s knowledge level fluctuates in the first
half of the learning process. It occurs because our system considers correct and incorrect responses during exercises.
When students frequently answer incorrectly, their knowledge state naturally does not improve. However, our EPPO
adjusts after a student answers multiple exercises incorrectly, recommending exercises that better match the student’s
current level. This adaptability to the student’s state allows our ALPN-EPPO system to assist each student in achieving
their learning goals.


We also examined the changes in the knowledge state of a single students=, i.e., the mastery level of various knowledge
components. When students aim to maximize their learning outcome towards 1, they must maximize their mastery
level of each knowledge component. On the other hand, when a student’s learning goals are more modest (e.g.,
β = 0.8), they can achieve their goals by improving the mastery of only a subset of concepts. However, we expect


9,000


7,500


6,000


4,500



3,000


1,500




|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
|||||||||||||||
|||||||||||||||
|||||||||||||||
||||||||||||ALPN-<br>ALPN-|PPO<br>PPO||
||||||||||||ALPN-|A2C|A2C|
|||||||||||||||



0.9


0.8


0.7


0.6


0.5


0.4


0.3



0

0 250 500 750 1,000 1,250 1,500 1,750 2,000 2,250 2,500 2,750 3,000


Episode


Figure 7: The cumulative reward during the agents’ training process.

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|Col10|Col11|Col12|Col13|Col14|Col15|Col16|Col17|Col18|Col19|Col20|Col21|Col22|Col23|Col24|Col25|Col26|Col27|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
||||||||||||||||||||||||||||
||||||||||||||||||||||||||||
||||||||||||||||||||||||Stud<br>Std|ent <br>t|C<br>||
||||||||||||||||||||||||u<br>Stud|en <br>ent|<br> E||
||||||||||||||||||||||||||||
||||||||||||||||||||||||||||
||||||||||||||||||||||||||||



0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24


Timestep (t)


Figure 8: The evolution of students’ knowledge level over time.


12


PRIME AI paper


algebra


analytic-geometry


arithmetic


calculus


geometry


logics


probability-statistics


Figure 9: The evolution of a student’s mastery level of each area during the learning process.


that after students have completed the learning path provided by the system, they will become more proficient in
each knowledge. To verify this, we present in Figure 9 the proficiency changes of a student for each area throughout
the learning process. The figure, from left to right, represents the student’s learning process. Each vertical column
represents the student’s proficiency level for each area at a specific moment, with darker colors indicating higher
proficiency. The figure shows that after completing this learning process, the student has significantly improved in
every area rather than just a few.


6 Conclusion


This paper presented an innovative approach to enhance E-learning systems through the Adaptive Learning Path Navigation (ALPN) system. By integrating the Attentive Knowledge Tracing (AKT) model with the newly proposed
Entropy-enhanced version of Proximal Policy Optimization (EPPO), the ALPN system generates adaptive learning
paths tailored to students’ knowledge states, thereby increasing the effectiveness of online learning.


The application of AKT, which employs an attention mechanism, accurately estimates a student’s knowledge state.
The EPPO model, on the other hand, optimizes the recommendation of learning materials, successfully outperforming
the traditional PPO model in our task by enhancing exploratory capabilities. The ALPN system harmonizes these
two models to facilitate highly adaptive and effective learning paths, offering the promising potential for adaptive
E-learning systems.


The performance of the ALPN system was tested against the existing Knowledge Tracing based Knowledge Demand
Model (KT-KDM). Our system demonstrated significant improvements, outperforming the KT-KDM method by 8.2%
on average to maximize students’ learning outcomes. Additionally, the ALPN system displayed superior diversity in
generating learning paths, with a 10.5% higher rate than the KT-KDM.


Even with the significant achievements demonstrated by the ALPN system, future work should further refine the
proposed approach. These improvements include tailoring the system to cater to a broader range of learning styles,
optimizing the model to increase further learning outcome improvements, and enhancing the model’s scalability to
accommodate even more significant numbers of students and diverse learning materials. Furthermore, exploring additional personalization parameters, such as learning style preferences or motivational factors, may prove beneficial in
creating even more tailored learning paths.


In conclusion, integrating AKT and EPPO in the proposed ALPN system marked a significant advancement in adaptive
E-learning. The system’s ability to generate reliable and adaptive learning paths enhances learning efficiency, making
E-learning an even more effective educational tool in the digital era. By continually seeking to refine and improve
such models, the education sector can stay at the forefront of technological advancements, ensuring quality learning
experiences for all students.


References


[1] S. Guri-Rosenblit. ‘distance education’ and ‘e-learning’: Not the same thing. High. Educ., 49(4):467–493, 2005.


[2] P. C. Sun, R. J Tsai, G. Finger, Y. Y. Chen, and D. Yeh. What drives a successful e-learning? an empirical
investigation of the critical factors influencing learner satisfaction. Comput. Educ., 50(4):1183–1202, 2008.


[3] B. Gilbert. Online learning revealing the benefits and challenges. Education Masters, 2015.


13


PRIME AI paper


[4] G. C. Oproiu. A study about using e-learning platform (moodle) in university teaching process. Procedia Soc.
Behav. Sci., 180:426–432, May 2015.


[5] N. S. Chen, Kinshuk, C. W. Wei, and H. J. Chen. Mining e-learning domain concept map from academic articles.
Comput. Educ., 50:1009–1021, 2008.


[6] P. Basu, S. Bhattacharya, and S. Roy. Online recommendation of learning path for an e-learner under virtual
university. In C. Hota and P. Srimani, editors, Distributed Computing and Internet Technology, volume 7753 of
Lecture Notes in Computer Science, pages 126–136. Springer Berlin Heidelberg, 2013.


[7] R. E. Mayer and R. Moreno. Nine ways to reduce cognitive load in multimedia learning. Educational Psychologist, 38(1):43–52, 2003.


[8] C. M. Chen, H. M. Lee, and Y. H. Chen. Personalized e-learning system using item response theory. Computers
and Education, 44(3):237–255, Apr 2005.


[9] A. Siren and V. Tzerpos. Automatic learning path creation using oer: A systematic literature mapping. IEEE
Trans. Learn. Technol., 15(4):493–507, 2022.


[10] T.-Y. Hsu, C.-K. Chiou, J. C. R. Tseng, and G.-J. Hwang. Development and evaluation of an active learning
support system for context-aware ubiquitous learning. IEEE Trans. Learn. Technol., 9(1):37–45, Jan./Mar. 2016.


[11] B. S. Bloom, M. D. Engelhart, E. J. Furst, W. H. Hill, and D. R. Krathwolh. Taxonomy of Educational Objectives:
The Classification of Educational Goals. Longman, London, U.K., 1956.


[12] J. S. Bruner. The Process of Education. Harvard University Press, Cambridge, MA, 1977. Original work
published in 1960.


[13] Y. L. Chi. Ontology-based curriculum content sequencing system with semantic rules. Expert Syst. Appl.,
36(4):7838–7847, May 2009.


[14] R. Gagne. The Conditions of Learning and Theory of Instruction. Holt, Rinehart, and Winston, 4 edition, 1985.


[15] J. A. Beane. Curriculum Integration: Designing the Core of Democratic Education. Teachers College Press,
New York, 1997.


[16] C. M. Reigleluth, editor. Instructional-Design Theories and Models: A New Paradigm of Instructional Theory
(Volume II). Lawrence Erlbaum Associate, Mahwah, NJ, USA, 1999.


[17] P. Karampiperis and D. Sampson. Adaptive learning resources sequencing in educational hypermedia systems.
Educ. Technol. Soc., 8(4):128–147, 2005.


[18] Z. Li, O. Papaemmanouil, and G. Koutrika. Coursenavigator: interactive learning path exploration. In Proc. 3rd
Int. Workshop Explor. Search Databases Web, pages 6–11. ACM, 2016.


[19] G. Durand, F. Laplante, and R. Kop. A learning design recommendation system based on markov decision
processes. In KDD- 2011: 17th ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2011.


[20] A. H. Nabizadeh, A. Mário Jorge, and J. Paulo Leal. Rutico: Recommending successful learning paths under
time constraints. In Proc. Adjunct Pub. 25th Conf. User Model., Adapt. Personalization, pages 153–158, 2017.


[21] M. M. Alian and R. Jabri. A shortest adaptive learning path in elearning systems: Mathematical view. J. Amer.
Sci., 5(6):32–42, 2009.


[22] H. Xie, D. Zou, F. L. Wang, T. L. Wong, Y. Rao, and S. H. Wang. Discover learning path for group users: A
profile-based approach. Neurocomputing, 254:59–70, Sep 2017.


[23] X. Feng, H. Xie, Y. Peng, W. Chen, and H. Sun. Groupized learning path discovery based on member profile. In
ICWL workshops, pages 301–310, 2010.


[24] A. H. Nabizadeh, J. P. Leal, H. N. Rafsanjani, and R. R. Shah. Learning path personalization and recommendation
methods: A survey of the state-of-the-art. Expert Syst. With Appl., 159, 2020.


[25] G. Abdelrahman, Q. Wang, and B. P. Nunes. Knowledge tracing: A survey. 2022.


[26] Y. LeCun, Y. Bengio, and G. Hinton. Deep learning. Nature, 521:436–444, May 2015.


[27] K. Arulkumaran, M. P. Deisenroth, M. Brundage, and A. A. Bharath. Deep reinforcement learning: A brief
survey. Signal Process. Mag., 34(6):26–38, Nov 2017.


[28] A. T. Corbett and J. R. Anderson. Knowledge tracing: Modeling the acquisition of procedural knowledge. User
Model. User-Adapted Interaction, 4(4):253–278, 1994.


[29] C. Piech, J. Bassen, J. Huang, S. Ganguli, M. Sahami, L. J. Guibas, and J. Sohl-Dickstein. Deep knowledge
tracing. In Advances in Neural Information Processing Systems NIPS, pages 505–513, 2015.


14


PRIME AI paper


[30] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polosukhin. Attention
is all you need. In Proc. 31st Int. Conf. Neural Inf. Process. Syst., pages 6000–6010, 2017.

[31] A. Ghosh, N. Heffernan, and A. S. Lan. Context-aware attentive knowledge tracing. In Proc. 26th ACM SIGKDD
Int. Conf. Knowledge Discovery and Data Mining, pages 2330–2339, Aug 2020.

[32] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. 2017.

[33] D. Cai, Y. Zhang, and B. Dai. Learning path recommendation based on knowledge tracing model and reinforcement learning. In Proc. IEEE 5th Int. Conf. Comput. Commun., pages 1881–1885, 2019.

[34] S. E. Embretson and S. P. Reise. Item Response Theory. Psychology Press, London, U.K., 2013.

[35] R. S. Sutton and A. G. Barto. Reinforcement Learning: An Introduction. MIT Press, 2018.

[36] C. J. Watkins and P. Dayan. Q-learning. Mach. Learn., 8(3-4):279–292, 1992.


[37] G. A. Rummery and M. Niranjan. On-line q-learning using connectionist systems. Technical report, University
of Cambridge, Department of Engineering, Cambridge, U.K., 1994.


[38] L. C. Baird. Residual algorithms: Reinforcement learning with function approximation. In Proc. Mach. Learn.,
pages 30–37, 1995.

[39] R. S. Sutton. Generalization in reinforcement learning: Successful examples using sparse coarse coding. In Proc.
Adv. Neural Inf. Process. Syst., pages 1038–1044, 1996.

[40] V. Mnih, K. Kavukcuoglu, D. Silver, A. A. Rusu, J. Veness, M. G. Bellemare, A. Graves, M. Riedmiller, A. K.
Fidjeland, G. Ostrovski, S. Petersen, C. Beattie, A. Sadik, I. Antonoglou, H. King, D. Kumaran, D. Wierstra,
S. Legg, and D. Hassabis. Human-level control through deep reinforcement learning. Nature, 518(7540):529–
533, 2015.


[41] V. Mnih et al. Playing atari with deep reinforcement learning. 2013.

[42] S. Levine, C. Finn, T. Darrell, and P. Abbeel. End-to-end training of deep visuomotor policies. J. Mach. Learn.
Res., 17(1):1334–1373, 2016.

[43] D. Bahdanau, K. Cho, and Y. Bengio. Neural machine translation by jointly learning to align and translate. In
ICLR, 2015.

[44] D. Silver et al. Mastering the game of go with deep neural networks and tree search. Nature, 529(7587):484–489,
2016.

[45] C. H. McGrath, B. Guerin, E. Harte, M. Frearson, and C. Manville. Learning gain in higher education. RAND
Corporation, Santa Monica, CA, 2015.

[46] T. P. Lillicrap, J. J. Hunt, A. Pritzel, N. Heess, T. Erez, Y. Tassa, D. Silver, and D. Wierstra. Continuous control
with deep reinforcement learning. 2015.

[47] J. Schulman, P. Moritz, S. Levine, M. Jordan, and P. Abbeel. High-dimensional continuous control using generalized advantage estimation. 2015.

[48] R. J. Williams. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Machine learning, 8:229–256, 1992.

[49] V. Mnih, A. P. Badia, M. Mirza, A. Graves, T. P. Lillicrap, T. Harley, D. Silver, and K. Kavukcuoglu. Asynchronous methods for deep reinforcement learning. 2016.

[50] H.-S. Chang, H.-J. Hsu, and K.-T. Chen. Modeling exercise relationships in e-learning: A unified approach. In
Proc. EDM, pages 532–535, 2015.

[51] L. Meng, W. Zhang, Y. Chu, and M. Zhang. Ld–lp generation of personalized learning path based on learning
diagnosis. IEEE Trans. Learn. Technol., 14(1):122–128, Feb. 2021.


15


