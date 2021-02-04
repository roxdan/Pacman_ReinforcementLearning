# analysis.py
# -----------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

# We changed the noise to a much lower ratio(take less random actions), this way the agent was able to reach the final reward using the optimal policy
def question2():
    answerDiscount = 0.9
    answerNoise = 0.01
    return answerDiscount, answerNoise


def question3a():
    answerDiscount = 0.3
    answerNoise = 0.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Prefer the close exit (+1), risking the cliff (-10)


def question3b():
    answerDiscount = 0.2
    answerNoise = 0.2
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Prefer the close exit (+1), but avoiding the cliff (-10)
# We set a low discount and noise but a high negative living reward, so the agent prefers the closer exit


def question3c():
    answerDiscount = 0.8
    answerNoise = 0.0
    answerLivingReward = 0.1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Prefer the distant exit (+10), risking the cliff (-10)
# We remove the noise for making optimal actions, set a high discount and a low living reward


def question3d():
    answerDiscount = 0.8
    answerNoise = 0.2
    answerLivingReward = -0.5
    return answerDiscount, answerNoise, answerLivingReward

    # If not possible, return 'NOT POSSIBLE'
# Prefer the distant exit(+10), avoiding the cliff(-10)
# Same as before but we set a higher penalty for living so he goes for the +10 exist, also low noise to avoid the cliffs


def question3e():
    answerDiscount = 0
    answerNoise = 0
    answerLivingReward = -1
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'
# Avoid both exits and the cliff (so an episode should never terminate)
# To achieve this we set both discount and noise to 0, so he never goes into the exit, also a high living reward


def question8():
    answerEpsilon = None
    answerLearningRate = None
    return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'


if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
