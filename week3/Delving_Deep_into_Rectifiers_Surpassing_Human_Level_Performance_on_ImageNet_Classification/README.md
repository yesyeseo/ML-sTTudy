# Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

[**ICCV 2015**](https://arxiv.org/pdf/1502.01852.pdf)

## Abstract

**정교한 activation unit(rectifiers)**가 neural network에 필수적이고, 그렇기 때문에 이 논문에서 두 가지 면을 통해 정교한 neural network에 대해 이야기함.

1. **PReLU**를 고안함: generalize the traditional rectified unit
    1. extra computational cost 거의 들지 않음 + overfitting risk 작음 → model fitting 개선
2. **Robust initialization method** 고안함: (우리가 아는 He Initialization!) 정교한 nonlinearities 고려함

## Introduction

이 논문 전의 recognition task는 다음 두 가지를 통해 성능을 개선함.

1. 더 강력한 model 자체를 build
2. overfitting을 방지하는 효율적인 strategies를 디자인

그런데 이에 따라 complexity가 증가(depth의 증가, width의 증가, 더 작은 stride의 사용, 새로운 nonlinear activations, 복잡한 layer design..)되어 NN은 training data에 점점 더 fitting하게 됨.

그리고 동시에 더 나은 generalization이 효율적인 정규화 techniques, 강력한 데이터 증강, 그리고 큰 스케일의 데이터에 의해 달성되기도 함.

 이러한 당시의 깊은 network의 달성에는 **ReLU**(Rectified Linear Unit)의 공도 있었음. 이것은 기존에 쓰이던 sigmoid보다 training 과정에서 더 converge를 촉진하고, 더 나은 결과를 도출하게 함. 그럼에도 불구하고 rectifier에 대한 고려는 model 설계 과정에서 충분히 이루어지지 않았음.

 그래서 이 논문에서는 rectifiers에 의해 도출되는 두 가지 좋은 면을 고려해서 NN을 뜯어봄. 

1. ReLU의 generalization된 형태의 activation function인 PReLU
    1. 이 activation function은 recitifers로부터 parameter를 learn함
    2. 그리고 추가적인 computational 비용 없이 정확도(성능)을 증가시킴
2. 매우 깊은 rectified model을 train하는 데에 어려움을 느낌 → nonlinearity 고려한 모델링 → initialization method를 고려함
    1. scratch 단계에서부터 시작되는 매우 깊은 모델의 convergence를 보다 쉽도록 도움
    2. 더 강력한 network architecture를 탐색하는 데에 flexibility를 부여함

## Approach

### Parametric Rectifiers

![Untitled](20220317_i%2096eaf/Untitled.png)

분류의 정확도를 높이기 위해 **parameter-free ReLU activation을 learned parametric activation unit로 바꿈**

**📌 Definition**

![Untitled](20220317_i%2096eaf/Untitled%201.png)

- yi: i번째 channel에서의 input of the nonlinear activation f
- ai: 음의 부분에서 slope를 컨트롤
    - ai == 0이 되면 그게 바로 ReLU
    - **ai이 learnable parameter가 되면 그것이 바로 PReLU**

→ 이 논문에서는 **channel마다 다른 nonlinear activation**을 가능하도록 함

- 만약 ai가 small & fixed한 값이라면:
    - ai == 0.01인 곳에서 LReLU
        - LReLU의 아이디어는 zero gradient를 방지하자는 것
        - 그런데 ReLU랑 비교했을 때 그렇게 큰 성능 향상이 이루어지지는 않음
    
    반면 PReLU는 전 모델에 걸쳐 adaptively하게 PReLU parameter를 학습함 → 더 specialized한 activation을 도출하기 위해 end-to-end training을 희망
    
- 아주 작은 수의 extra parameter을 요구
    - 이 수는 channel의 수와 같음 → weight의 수에 비하면 무시할 수 있는 값
        - 그래서 overfitting의 리스크는 없다고 봄 **(근데 이게 무슨 인과관계인지는..)**
    - channel-shared variant 고안
        - coefficient는 one layer의 모든 channel에 대해 공유됨
        - 그래서 각 layer에 대해 오직 하나의 추가적인 parameter만 요구함

**📌 Optimization**

PReLU는 backpropagation을 통해 trian되고, other layers와 동기적으로 optimized됨.

![Untitled](20220317_i%2096eaf/Untitled.jpeg)

- ai update에 고려/이용되지 않은 것
    - **weight decay:** weight decay는 ai가 0이 되도록 강제하기 때문에 ReLU랑 비슷해짐
    - **regularization:** 안 사용해도 1을 넘어가지 않았음
    - **constrain of the range:** non-monotonic(단조롭지 않도록)한 activation function이 되도록 ai의 범위를 제한하지 않음

**📌 Comparison Experiments**

- 깊고 효율적인 14 weight layer model에 대해서 비교 진행
    
    filter size, filter number list는 table 1 참고
    
    ![Untitled](20220317_i%2096eaf/Untitled%202.png)
    
- **baseline**: ReLU를 conv layer와 첫 2개의 fc layer에 적용시킴
    - ImageNet 2021 10-view testing에 대해
        - top-1 errors: 33.82%
        - top-5 errors: 13.34%
- **compare**: ReLU를 PReLU로 바꿔서 scratch단계부터 다시 train시킴
    - top-1 error는 32.64%로 1.2% 더 개선됨

![Untitled](20220317_i%2096eaf/Untitled%203.png)

 위 table 2에서는 channel-wise/channel-shared PReLU도 성능적으로 비교할 수 있다는 점을 시사함. 

- channel-shared version
    - ReLU에 비해 13개의 parameter만 더 생김
    - 그런데 이런 작은 차도 1.1% 정도의 성능 개선에 크게 기여함
    - 이런 점은 activation function에서의 adaptively learning의 중요성을 보여 줌

 그리고 이에 대해 table 1에서 관찰되는 두 가지 흥미로운 점이 있음

1. conv1: coefficients 0.681, 0.596: 모두 1보다 확실히 큼
    1. conv1은 edge, texture dectector → **positive, negative 성질 모두** 보존됨
    2. 필터 수가 제한되어 있는 경우 효율적임
2. channel-wise version에서 더 깊은 conv layer는 작은 coefficients를 가짐
    1. activation이 깊이가 깊어질수록 더 nonlinear해졌다는 것을 의미함 (왜냐면 양수일 때의 scope가 1이므로)
    2. 그러니까 **깊이가 깊어질수록 전 단계의 정보를 더 keep**하는 경향이 있다는 의미

### Initialization of Filter Weights for Rectifiers

Recitifer network는 기본적으로 sigmoid-like activation network보다는 train하기 쉽지만 나쁜 initizalization은 non-linear한 system의 learning을 막을 수 있음. 그래서 아주 깊은 네트워크 train의 걸림돌을 제거하도록 **robust**한 initialization method를 고안하고자 함.

- 기존의 initialization
    - 가우시안 분포에 의한 random weights drawn
    - fixed standard deviations → 깊은 model들은 converge에 어려움을 가짐
        - 첫 8개의 layer에 대해 pre-train → deeper model을 initialization
        - 그러나 train time이 많이 들고 poorer local optimum을 도출시킴
        - auxiliary classification이 중간 layer의 converge를 돕기 위해 추가되기도..
    - **xavier**
        - **activation이 linear하다는 가정 아래 만들어짐**
        - ReLU나 PReLU 같은 nonlinear한 activation에는 부적절함
        - 그리고 엄청 deep한 모델에는 부적절함
        

**📌 Forward Propagation Case**

 each layer에서의 response의 **variance**를 보는 것이 일단 main idea

- conv layer에서의 response:
    - *x*: *(k^2)c-by-1* vector
        - c input channel에 대한 kxk개의 pixel 표현
        - n = (k^2)c : response의 connection의 숫자
    - *W: d-by-n* matrix
        - d: W row의 filter의 갯수
    - *b*: vector of biases
    - *y*: output map의 pixel에서의 response
    - *l*: layer의 index

![Untitled](20220317_i%2096eaf/Untitled%204.png)

- xl = f(y_(l-1))에서 f가 activation
- cl = d_(l-1)

*Wl의 초기화된 요소들*: independent & share the same distribution (Xl are also)

그리고 Wl, Xl: independent of each other

그렇게 해서:

(참고: [https://medium.com/@shoray.goel/kaiming-he-initialization-a8d9ed0b5899](https://medium.com/@shoray.goel/kaiming-he-initialization-a8d9ed0b5899))

![Untitled](20220317_i%2096eaf/Untitled%205.png)

![Untitled](20220317_i%2096eaf/Untitled%206.png)

yl, xl, wl은 각각의 random variable을 나타냄. wl은 zero mean을 가지고 있음. 그리고 **independent한 variable의 곱**이므로 다음 식으로 나타낼 수 있음.

![Untitled](20220317_i%2096eaf/Untitled%207.png)

![Untitled](20220317_i%2096eaf/Untitled%208.png)

 여기서 E는 기댓값. xl의 평균이 0이 아닌 이상 E[xl^2] ≠ Var[x1]이라는 점을 유의해야 함. 

ReLU에서는 xl = max(0, y_(l-1))이고 이것은 zero mean을 가질 수 없음. 

만약 w_(l-1)이 0을 중심으로 대칭적인 분포를 가진다면, y_(l-1) 또한 zero mean을 가지며 0을 중심으로 대칭적인 분포를 가진다. 이는 f가 ReLU일 때

![Untitled](20220317_i%2096eaf/Untitled%209.png)

라는 결론을 도출시키고, 이를 위 식에 대입시키면

![Untitled](20220317_i%2096eaf/Untitled%2010.png)

이러한 식을 도출시킬 수 있다. 이것에 대해 L개의 layer가 함께 진행되면: 

![Untitled](20220317_i%2096eaf/Untitled%2011.png)

이 식을 얻는다. 

이 분산은 initialization 설계의 핵심인데, **proper initialization은 input signal을 엄청 감소시키거나 혹은 증가시키는 것을 피해야 하기 때문에** proper scalar을 설정해야 함.

![Untitled](20220317_i%2096eaf/Untitled%2012.png)

 그래서 이게 가장 적절하고, 이는 std가 (2/nl)**(1/2)인 가우시안 분포를 따름. 이것이 initialization method. 그리고 l=1일 때는 input에 ReLU가 적용되지 않으므로 n1Var[w1] = 1임. 근데 1/2라는 factor는 첫째 레이어에서 하든지 말든지 별 상관이 없고 여기서는 적용시킨다고 함. (all)

***plus) python에서의 구현..***

![Untitled](20220317_i%2096eaf/Untitled%2013.png)

**📌 Backward Propagation Case**

(계산을 거쳐..)

결론: 

![Untitled](20220317_i%2096eaf/Untitled%2012.png)

**📌 Discussions**

- PReLU에서는

![Untitled](20220317_i%2096eaf/Untitled%2014.png)

이렇게 초기화시키고, a는 coefficient의 initialized value임

**a = 0이면 ReLU case와 동일해짐.**

**a = 1이면, linear case와 같아짐.**

**📌 Comparisons with “Xavier” Initialization**

- Xavier:

![Untitled](20220317_i%2096eaf/Untitled%2015.png)

- He:

![Untitled](20220317_i%2096eaf/Untitled%2016.png)

가장 큰 차이점은 **nonlinearlity**에 대한 접근. Xavier의 derivation은 linear case에 대해서만 고려함. 

Xavier은 가우시안 분포에서 std가 (1/nl)**(1/2)로 표현되는데, 이것의 std는 He의 std의 (1/(2)**(L/2))가 됨. **이것은 converge하기에는 너무 작음**. 

![Untitled](20220317_i%2096eaf/Untitled%2017.png)

22개의 layer가 있을 때는 둘 다 converge하기는 함. 그러나 He가 조금 더 빨리 error을 감소시키고 converge함. 그리고 정확도도 He가 더 나았음. 

![Untitled](20220317_i%2096eaf/Untitled%2018.png)

좀 더 깊은 모델에서는(16 conv layer, 256 2x2 filters) 이제 30-layer 모델인데, Xavier는 stalls함. The gradients가 사라짐(gradient diminishing)

## Experiments on ImageNet

- Comparisons between ReLU and PReLU: PReLU가 computional cost의 증가 없이 small/large model 모두에서 accuracy를 improve시킴

![Untitled](20220317_i%2096eaf/Untitled%2019.png)

- Comparisons of Single-model Results & Multi-model Results:

![Untitled](20220317_i%2096eaf/Untitled%2020.png)
