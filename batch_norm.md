# He, batch normalization

기간: 2022년 3월 18일 → 2022년 3월 24일
주차: 5

# 1. Introduction

### mini-batch의 장점

- mini-batch로부터 얻은 gradient는 training set의 gradient의 추정치 값이 될 수 있고 minibatch 크기가 커질수록 성능이 좋아진다.
- 현대 computing platform에 의해 병렬처리가 가능해졌고 이 덕분에 m번의 계산을 각각하는것보다 batch만큼의 연산을 한번에 하는게 더 효율적이다.

stochastic gradient는 간단하고 효율적이지만 model의 하이퍼 파라미터(특히 lr, 파라미터 초기값)를 잘 지정해줘야 한다. 그리고 training은 이전의 모든 layer의 파라미터에 영향을 받기 때문에 network가 깊어질 수록 network 파라미터 변화량이 증폭된다.

layer의 input 분산의 변화가 문제가 되는 이유: layer들이 새로운 분산에 맞게 연속적으로 조절되기 때문

### batch normalization의 이점

- 모델의 파라미터와 초기값의 의존성을 낮출 수 있다
- 발산 걱정없이 학습률을 크게 할 수 있어 training 속도를 빠르게 할 수 있다.
- Regularization도 가능하게 해주어 dropout을 할 필요 없게 해준다.
- saturated mode에 빠지지 않게 해주어 sigmoid와 같은 saturated nonlinearity를 사용할 수 있게 해준다.

# 2. Towards Reducing Internal Covariate Shift

internal covariate shift는 training동안 network parameter 값이 변해서 network activation의 분산이 변하는 것을 말한다. 이는 training 속도를 느리게 한다.

원래 internal covariate shift는 input의 분산이 변하는 것을 말하는데 network의 일부분에 covariate shift가 적용되도 learning system 전체에까지 영향을 주기 때문에 이는 중요한 문제가 된다. 

input이 whitened(정규분포 모양)일때 training 수렴 속도가 빠르다는 유명한 발견을 사용한다. 그래서 각 layer의 input을 whitening함으로써 분산을 수정하고 internal covariate shift 문제를 줄이겠다는 아이디어가 고안된다.

# 3. Normalization via Mini-Batch Statistics

각 layer들의 input에 full whitening하는 것은 비용이 많이 들고 항상 미분가능한 것도 아니어서, 두가지 단순화를 한다.

**1. 네트워크에 삽입된 tranformation은 identity transform으로 나타낼 수 있다.**

layer의 input과 output의 feature에 대해 같이 whitening하는 것이 아니라 각 scalar feature를 각각 평균 값을 0으로, 분산 값을 1로 만들어 독립적으로 normalize한다. layer가 d차원의 input x를 가진다고 하면 각 타원에서의 normalize식은 다음과 같이 나타낼 수 있다.

![스크린샷 2022-03-24 오후 3.30.45.png](He,%20batch%20%20a7a20/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_3.30.45.png)

이때 기댓값과 분산은 training data에대한 값이다.(LeCun et al(1998b)에 따르면 이러한 normalization은 feature들이 서로 상관 관계(두 변수가 선형적인 관계를 가지고 있음)에 있더라도 수렴 속도를 빠르게 한다는 것이 증명되었다)

normalizing이 layer의 각 input이 단순히 layer가 나타내는 것을 바꿔줄 수 있다는 점을 다루기 위해 필요한 단순화이다.

이 단순화를 실현하기 위해 각 activation $x^{(k)}$에 대해 parameter 쌍 $γ^{(k)}, β^{(k)}$를 도입하여 normalized value `y^(k)` 의 크기를 조절하고 shift한다. ($x̂=$**Norm**$(x,X)$)

$$
y^{(k)}=γ^{(k)}x̂^{(k)}+β^{(k)}
$$

위의 gamma 값과 beta 값의 파라미터는 original model parameter와 함께 학습되고 representation power에 재저장된다. 실제로는 gamma값은 x^(k)의 표준편차 값으로, beta값은 x^(k)의 기댓값으로 셋팅됨으로써 optimal 값을 찾았다고 가정할 때 원래의 activation을 복원할 수 있음.

**2. 각 mini-batch는 각 activation의 평균과 분산(에 거의 가까운)추정치를 만들어낸다.**

전체 training set을 기반으로 각 training step에 대해 normalize activation을 사용되는 것을 보여줘야 하는데 stochastic optimization을 사용하는 경우 이는 너무 실용적이지 않기때문에 이 단순화를 사용한다. 

이로써 gradient backpropagation에서 완전한 normalization의 사용을 보여줄 수 있다. 단, mini-batch는 공분산보다는 각 차원의 분산을 사용한다.(공분산을 사용하게 되면 mini-batch 크기가 whitened될 activation보다 작을 가능성이 있어서 singular 분산 행렬의 결과를 내놓을 수 있기 때문에 regularization을 해줘야한다.)

모든 차원에서의 연산은 동일하므로 차원에 대한 정보를 생략하면 batch normalization transform의 알고리즘은 다음과 같다.

![스크린샷 2022-03-24 오후 4.00.59.png](He,%20batch%20%20a7a20/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.00.59.png)

ε은 mini-batch의 분산을 numerical 안정화를 위해 조절하기 위해 더해진 상수값이다.

BN transform에서 γ값과 β값은 같은 training example과 mini-batch내의 다른 example에도 영향을 받는 값이고 때문에 두 파라미터를 적용한 값 y값은 다른 layer로 전달된다. 

각 mini-batch가 같은 분산으로부터 추출되고 엡실론 값을 무시한다고 하면 어느 $x̂$에 대해서 기댓값은 0, 분산값은 1이 된다. 각 평준화된 activation인 $x̂^{(k)}$는 선형 transform인 

![스크린샷 2022-03-24 오후 4.14.31.png](He,%20batch%20%20a7a20/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.14.31.png)

의 sub-network으로의 input으로 볼 수 있고 뒤이어 다른 processing인 original network까지 끝낼 수 있다. 이 sub-network의 input들의 평균과 분산은 모두 수정되고 이러한 평준화된 activation이 training과정에서 공분산값이 변하더라도 평준화된 input을 도입한 것은 sub-network, 결과적으로는 network 전체의 training속도를 가속화한다.

training하는 동안 이 transformation을 통해 얻은 loss값의 gradient를 backpropagation 해야하고 BN transform의 파라미터에 대한 gradient값을 계산해야 한다. 이는 아래와 같이 연쇄법칙을 사용하여 계산해낼 수 있다.

![스크린샷 2022-03-24 오후 4.19.05.png](He,%20batch%20%20a7a20/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-24_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.19.05.png)

따라서 BN transform은 평준화한 activation을 network에 도입한 미분가능한 transformation이 된다. 이것은 모델을 training할 때 layer들이 계속해서 input 분산을 적은 internal covariate shift를 가지는 상태에서 학습할 수 있다는 것을 보장한다. 게다가 learned affine transform은 이러한 평준화된 activation이 BN transform이 identity transformation으로 나타낼 수 있고 network capacity를 유지할 수 있도록 해준다.