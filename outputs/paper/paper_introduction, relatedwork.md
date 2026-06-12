```latex
\section{Introduction}

대화 시스템이 수십 turn 이상의 장기 상호작용으로 확장됨에 따라, 누적된 대화 흐름을 의미적으로 일관된 토픽 단위로 추적하는 대화 토픽 분절(Dialogue Topic Segmentation, DTS)이 회의 요약~\citep{he2025one}, 장기 대화 에이전트의 메모리 구성~\citep{pan2025memory}, proactive dialogue 시스템~\citep{zhang2025proactive} 등 대화 시스템의 전처리층 역할을 하고 있다. 그런데 한 가지 묘한 어긋남이 있다 — 이러한 응용은 본질적으로 실시간 환경에서 구동되어야 하지만, DTS 연구는 거의 대부분 대화가 끝난 후 전체를 본 상태에서 한 번에 분절하는 오프라인 가정 아래 발전해 왔다~\citep{he2025one, pan2025memory, zhang2025proactive}. 이 가정은 실시간 환경에서 세 가지 제약을 가진다: (i) 미래 발화를 알 수 없고, (ii) 턴 단위 latency 예산을 지켜야 하며, (iii) 대화가 길어질수록 계산 비용이 커진다. ****

이러한 어긋남을 메우려는 시도로 \citet{lin2023topic, lin2023multi}는 응답을 포함한 미래 발화에 접근할 수 없는 response-unknown 세팅에서 실시간으로 토픽 전환을 감지하는 과제를 Dialogue Topic Shift Detection(DTSD)으로 명명하며 이를 진정한 온라인 평가의 기준으로 제시하였다. 그러나 정작 실험은 다음 응답까지는 허용하는 response-known 세팅에서 수행되어 명명과 평가 사이에 괴리를 남겼고, 후속 연구에서도 엄밀한 response-unknown 평가와 실제 시스템 응용을 좌우하는 latency 분석은 충분히 다루어지지 않았다. 따라서 매 턴 과거 발화만으로 즉시 결정을 내리는 진정한 future-blind 체제에서 DTSD가 어디까지 도달할 수 있는지는 여전히 열린 문제이다.

흥미롭게도, 사람의 뇌는 자연 경험을 실시간으로 분절할 때 동일한 제약 아래 작동한다. 사건 분절 이론(Event Segmentation Theory; \citealp{zacks2007event,zacks2009segmentation})에 따르면 뇌는 다음에 무엇이 올지를 끊임없이 예측하며, 그 예측이 빗나가는 순간(prediction error)을 사건의 경계로 인지한다. 이 메커니즘에서 본 논문이 online-DTS로 가져오는 설계 원리는 두 가지다. 

\textbf{원리 1: 예측 오류는 여러 시간 척도에서 동시에 누적된다.} \citet{geerligs2022partially}는 뇌가 사건을 분절할 때 짧은 시간 척도(직전 자극 대비)와 긴 시간 척도(누적된 맥락 대비)의 예측 오류를 동시에 추적함을 fMRI로 보였다. 어느 한 척도만 보면 잡음에 속거나 진짜 경계를 놓친다.

\textbf{원리 2: 경계의 강도 정보는 결정과 함께 보존된다.} 해마는 사건 경계에서 반응하되, 경계가 뚜렷할수록 더 큰 반응을 낸다~\citep{baldassano2017discovering, ben2018hippocampal} — 결정과 함께 보존된 강도 정보가 downstream에서 경계 강도에 따른 처리를 가능하게 한다.

본 논문은 매 시점마다 이중 시간 척도의 예측 오류를 누적하고 그 강도를 함께 출력하는 온라인 알고리즘 Hi-OnTop를 제안한다. 이 효과를 검증하기 위해 DTS 표준 벤치마크에서 평가하며, 장기 대화 메모리 시스템에서의 plug-in 검증을 수행한다. Our contributions are listed as follows:

\begin{itemize}

\item \textbf{Hi-OnTop 프레임워크 제안:} 뇌 인지 과정을 영감 삼아 라벨이나 미래 정보 없이 가볍게 동작하는 프레임워크를 제안하며, TIAGE · Dialseg711 · SuperDialseg에서 ~만큼의 성능 향상을 보인다(Score [TBD]).

\item \textbf{장기 대화 메모리 시스템에서의 plug-in 검증:} SeCom 파이프라인의 분절 백엔드를 Hi-OnTop로 교체하여, 베이스라인 대비 seg latency가 turn당 $O(N) \to O(1)$로 감소하면서 다운스트림 retrieval 품질이 효과적으로 유지됨을 보인다.

\end{itemize}

\section{관련 연구}
\subsection*{Dialogue Topic Segmentation}

DTS 연구는 어휘 또는 주제 응집도에 기반한 비지도 알고리즘에서 출발한다~\citep{hearst1997texttiling, glavas2016graphseg}. 이들 방법은 리소스 제약 환경에서 신속한 프로토타이핑이 가능하다는 장점이 있으나, 의미 표상의 부재로 인해 성능 향상에 한계를 보였다. 이후 BERT 등과 같은 사전학습 언어모델 임베딩과 결합한 비지도학습 계열이 의미 표상력을 보강하며 등장하였다~\citep{xing2021csm}. 한편 지도학습 계열에서는 SuperDialseg와 같은 라벨 코퍼스 위에서 BERT 또는 T5 인코더를 직접 미세조정하여 특정 도메인에서 최고 수준의 정확도를 달성하였으나, 라벨 비용과 도메인 외 일반화 한계를 동반한다~\citep{liu2019roberta, ups2025}. 최근에는 라벨 의존성을 우회하기 위해 분절 결정 자체를 대형 언어모델 프롬프팅에 위임하는 흐름이 자리잡고 있다~\citep{dashdts2025, defdts2025}. 그러나 이러한 접근들은 여전히 결정 시점에 후보 경계의 양측 발화가 모두 가용하다는 오프라인 가정 위에 서 있어, 매 턴 과거 발화만으로 즉시 결정을 내려야 하는 온라인 체제에 그대로 이식되기 어렵다.

\subsection*{Dialogue Topic Shift Detection}

이러한 한계에 대응하여 ~\citep{lin2023topic, lin2023multi}는 미래 발화에 아예 접근할 수 없는 response-unknown 세팅과 다음 응답까지의 접근까지는 허용하는 response-known 세팅을 구분하고, 다음 응답을 차단하는 진정한 future-blind 평가의 필요성을 제기하였다. 이들은 대화 시스템에서 response-unknown 세팅으로 실시간 토픽 전환을 감지하는 과제를 Dialogue Topic Shift Detection(DTSD)으로 명명하였다. 이들이 제안한 교사-학생 방법론은 추론 시 response-unknown으로 동작하지만, 특정 언어·도메인에 한정되어 있고 latency 절감이나 실배포 가능성 측면의 고려가 부재하다.

본 연구는 인접 발화 간의 짧은 시간 척도에서 나타나는 표면적 변화와 긴 시간 척도의  변화를 동시에 추적하는, 뇌 인지과정에 기반한 DTSD 파이프라인인 Hi-DTSD를 제안한다. 이는 calibration 기법을 통해 토픽 전환 라벨이 없는 데이터에서도 매 턴 response-unknown 세팅에서 즉시 전환을 감지할 수 있도록 대화 도메인 의존성을 낮춘다. 또한 기존 offline DTS의 추론 프레임워크를 과거 발화만으로 구성된 시계열 위에서 평가할 수 있도록 online DTSD 형태로 재구성함으로써, 실제 사용자 관점의 turn-level latency를 segmentation 성능과 함께 측정하여 downstream task 평가까지 확장한다.
```