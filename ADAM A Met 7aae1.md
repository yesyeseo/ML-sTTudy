# ADAM: A Method for Stochastic Optimization

# Concepts

(참고: [https://light-tree.tistory.com/140](https://light-tree.tistory.com/140), [https://koreanfoodie.me/178](https://koreanfoodie.me/178))

## Momentum

![스크린샷 2022-03-17 오후 4.26.51.png](ADAM%20A%20Met%207aae1/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.26.51.png)

momentum은 gradient descent 기반의 optimization algorithm이다. 위 식은 momentum의 식이다. v는 일종의 가속도의 개념으로 이해할 수 있다. v로 인해 weight 파라미터는 가중치가 감소하던 방향으로 더 감소하거나, 증가하던 방향으로 더 증가하게 된다. v값은 0으로 초기화된다.

참고로 moment와 momentum은 다른 개념이다.

## AdaGrad

학습률을 조절하여 최적의 파라미터를 찾는 optimization algorithm이다. 학습률 감소(learning rate decay)를 통해 학습률을 조절하게 된다.

![스크린샷 2022-03-17 오후 4.30.38.png](ADAM%20A%20Met%207aae1/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.30.38.png)

학습률 감소는 말그대로 학습을 진행하면서 학습률을 점차 줄여가는 방법이다. 학습률은 너무 작으면 학습시간이 오래걸리고, 너무 크면 학습이 제대로 이루어지지 않기 때문에 loss 값이 큰 처음에만 학습률을 크게하여 빠르게 최적의 파라미터에 가까워지도록 학습하고, 점점 학습률을 줄여 천천히 최적의 파라미터 값에 가까워지도록 학습한다. 

위의 수식이 학습률을 조절하는 식인데, 학습률을 조절하는 값인 h를 사용하여 학습률을 점차 줄여나간다. h는 현재 loss function의 기울기를 반영하여 기울기가 크면 더 많이 학습률을 작게하고, 기울기가 작으면 학습률을 덜 작게만든다. 이렇게 당시의 기울기에 따라 학습률이 조절되는 정도가 달라지는 것을 보고 매개변수에 적응적으로 학습률을 조절한다고 하여 adaptive gradient이어서 AdaGrad라고 한다. 

## RMSProp

AdaGrad는 과거의 기울기를 제곱하여 계속 더하면서 학습하기 때문에 학습을 진행할수록 갱신되는 정도가 약해진다. (위의 식에서 h값이 계속 증가하니까 학습률에 곱해지는 값인 1/sqrt(h)값은 점점 감소한다.) 실제로 무한히 계속 학습한다면 갱신량은 0에 수렴하게 되어 전혀 갱신되지 않게 된다. RMSProp은 이 문제를 개선하기 위해 고안된 최적화 알고리즘이다.

![t=i 일때의 RMSProp의 update 수식](ADAM%20A%20Met%207aae1/%E1%84%89%E1%85%B3%E1%84%8F%E1%85%B3%E1%84%85%E1%85%B5%E1%86%AB%E1%84%89%E1%85%A3%E1%86%BA_2022-03-17_%E1%84%8B%E1%85%A9%E1%84%92%E1%85%AE_4.53.48.png)

t=i 일때의 RMSProp의 update 수식

RMSProp은 과거의 계속해서 모든 기울기를 균일하게 더하는 것이 아니라 오래된 기울기는 잊고 새로운 기울기 정보를 많이 반영한다. 이를 지수이동평균(exponential moving average, EMA)라고 하고, 과거 기울기의 반영 규모를 기하급수적으로 감소시킨다.

# **Intro**

Adam은 gradients의 1차와 2차 moment의 추정치로부터 다른 파라미터들에 대해 각각 adaptive한 learning rate를 계산한다. 이는 AdaGrad와 RMSProp의 장점을 결합하여 sparse gradients에도 잘 적용되고, on-line이나 non-stationary한 환경에서도 잘 작동된다. Adam의 중요한 이점은 gradient의 변화에도 파라미터의 갱신량이 변함이 없다는 것과, stepsize 하이퍼 파라미터에 의해 stepsize가 대략 제한된다는 것, non-stationary objective이나 sparse gradients에서도 잘 동작한다는 것이다. 

# Algorithm

![Untitled](ADAM%20A%20Met%207aae1/Untitled.png)

f(theta)은 noisy objective function(dropout과 같은 data subsampling을 적용한 loss function)을 말한다. 

알고리즘은 m_t로 표현되는 기울기의 EMA(exponential moving average)과 v_t로 표현되는 기울기의 제곱의 EMA을 update한다. 이 때 하이퍼 파라미터 beta_1과 beta_2(0과 1사이의 실수 값)는 이런 EMA의 decay rate을 조절한다. 두 EMA(m_t와 v_t)는 각각 1차 moment(기댓값)와 2차 raw moment(uncentered 분산)의 추정치이다. 

t=0일때, decay rate이 매우 작은 값일때(beta가 1에 가까울때)와 같은 경우 bias가 0이 되도록 moment 추정하기 위해서 m_t와 v_t의 값을 0으로 초기화해야한다. 이런 초기화의 편향 문제는 해결하기 쉽고 그렇게 해서 수정된 편향 추정치 값은 m_t’(m_t hat)과 v_t’(v_t hat)로 표기한다.

이 뒤에 Adam의 update 규칙, 초기화 편향 수정, convergence 분석에 대해 더 자세하게 수식을 들어 설명하였는데 이해를 아직 다 못하였고 이런 부분까지 세세하게 알 필요 없을 것 같아서 제외하고 Adam 알고리즘에 대해 이해한 바를 간략하게 설명하면 

Adam은 기울기(m_t)와 기울기 제곱한 값(v_t)을 사용하여 갱신량(stepsize)를 조절하는데, 이 때 m_t와 v_t의 값을 크게 바뀌지 않도록(bias가 크지 않도록) decay rate관련 하이퍼 파라미터인 beta값을 사용하여 EMA(기울기 변화의 관성을 따라가지만, 오래된 기울기 혹은 기울기 제곱의 값은 반영하지 않고 최근의 값을 많이 반영하기 위해)를 구하여 파라미터를 학습한다.

이게 왜 다른 최적화 알고리즘들보다 획기적으로 좋은 이유는 아직 파악하지 못했다. 알게되면 이후 글을 수정할 계획.

# Related Work

Adam은 RMSProp과 AdaGrad와 직접적으로 관련이 있다. vSGD, AdaDelta, natural Newton method와 같은 다른 stochastic optimization은 모든 stepsize를 곡률의 1차식의 정보만을 사용하여 값을 설정한다. Sum-of-Functions Optimizer인 SFO는 미니배치에 기반을 둔 quasi-Newton method이지만 GPU와 같이 메모리 제한된 시스템에서도 종종 실현가능하도록 dataset의 미니배치 개수가 선형적으로 증가하는 메모리가 요구된다. Adam은 t에서의 update된 가속도의 값이 Fisher information matrix의 diagonal로 근사값이기 때문에 Adam은 natural gradient descent처럼 data의 geometry를 조절하도록 preconditioner를 사용한다. 그러나 Adam의 preconditioner는 diagonal Fisher information matrix approximation의 inverse의 제곱근으로 preconditioning함으로써 AdaGrad의 것처럼 vanilla NGD보다는 값이 더 유지되면서 조절되는 편이다. 

### RMSP**rop**

momentum과 사용되는 버전이 많이 사용된다. momentum을 사용한 RMSProp과 Adam의 몇 가지의  중요한 차이점은:

- momentum을 사용하는 RMSProp은 rescaled gradient에 momentum을 사용하여 parameter를 update하는 반면, Adam은 gradient의 1차와 2차 momentum의 평균을 사용하여 parameter를 update한다.
- RMSProp은 bias-correction term이 없다: 이는 beta_2가 1에 가까워질때 주로 문제가 되는데, correcting bias는 매우 큰 stepsize를 야기하고 종종 발산을 야기한다. section 6.4에서 증명함

### AdaGrad

이 알고리즘은 sparse gradient에 잘 적용된다.  이 부분은 아직 이해 못함