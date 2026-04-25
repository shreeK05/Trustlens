import { useEffect, useRef, useState } from "react";
import type { FormEvent } from "react";
import {
  ArrowRight,
  BarChart3,
  BadgeCheck,
  BrainCircuit,
  CheckCircle2,
  ChevronRight,
  Clock3,
  FileSearch,
  Layers3,
  MessageCircleMore,
  Search,
  ShieldCheck,
  Sparkles,
  Star,
  Store,
  TrendingUp,
  TriangleAlert,
  Zap,
  MessageSquareWarning,
} from "lucide-react";

type FakeReviewResult = {
  fake_count: number;
  genuine_count: number;
  total_analyzed: number;
  fake_probability: number;
  avg_fake_score: number;
  authenticity_score: number;
  flagged_reviews: { index: number; text: string; fake_probability: number }[];
};

type SentimentResult = {
  positive: number;
  neutral: number;
  negative: number;
  overall: string;
  sentiment_score: number;
  mismatch_detected: boolean;
  mismatch_reason: string;
  sentiment_distribution: { positive: number; neutral: number; negative: number };
};

type PriceAnomaly = {
  is_anomaly: boolean;
  anomaly_score: number;
  price_trend: string;
  discount_pct: number;
  vs_avg_history: number;
  confidence: number;
};

type SellerRisk = {
  risk_level: string;
  risk_score: number;
  confidence: number;
  probabilities: { low: number; medium: number; high: number };
  color: string;
  feature_importance: Record<string, number>;
};

type AnalysisResult = {
  title: string;
  price: number;
  mrp: number;
  discount: number;
  image: string;
  seller: string;
  brand: string;
  category: string;
  rating: string;
  reviews: string;
  features: string[];
  price_history: { month: string; price: number }[];
  score: number;
  grade: string;
  verdict: string;
  trust_probability: number;
  confidence_pct: number;
  certificate: string;
  shap_contributions: Record<string, number>;
  risk: string;
  pros: string[];
  cons: string[];
  fake_reviews: FakeReviewResult;
  sentiment: SentimentResult;
  price_anomaly: PriceAnomaly;
  seller_risk: SellerRisk;
  analysis_time_s: number;
  ml_powered: boolean;
  source_mode: string;
  analyzed_at: string;
};

type ModelStats = {
  models: Record<
    string,
    {
      algorithm: string;
      accuracy?: number;
      detection_rate?: number;
      type: string;
      features: string;
    }
  >;
  pipeline: string;
  total_models: number;
};

const API_BASE = import.meta.env.VITE_TRUSTLENS_API_URL ?? "http://127.0.0.1:8000";

const heroFacts = [
  "Live Amazon URL scanning",
  "5-model trust pipeline",
  "Explainable verdicts",
  "Local backend inference",
];

const featureCards = [
  {
    icon: Search,
    title: "Review authenticity",
    text: "Detects copy-paste spam, unnatural repetition, and review bursts with a trained NLP model.",
  },
  {
    icon: MessageCircleMore,
    title: "Sentiment mismatch",
    text: "Flags products where star rating and review language do not tell the same story.",
  },
  {
    icon: TrendingUp,
    title: "Price anomaly watch",
    text: "Checks price history and discount behavior for suspicious drops or unstable pricing.",
  },
  {
    icon: Store,
    title: "Seller risk scoring",
    text: "Ranks seller profiles with a multi-class XGBoost classifier and confidence breakdown.",
  },
  {
    icon: Layers3,
    title: "Trust ensemble",
    text: "Combines model signals into a final trust grade with a single score the user can understand.",
  },
  {
    icon: ShieldCheck,
    title: "Clear verdicts",
    text: "Shows what pushed the score up or down, so decisions are visible instead of hidden.",
  },
];

const storyCards = [
  {
    title: "Built for fast checks",
    text: "Paste a product link, get a trust score, and inspect the reasons behind it in a few seconds.",
  },
  {
    title: "Designed for trust",
    text: "The interface emphasizes clarity and confidence instead of cluttered technical dashboards.",
  },
  {
    title: "Grounded in real data",
    text: "The backend uses real Amazon-related datasets and returns live analysis or a clear error.",
  },
];

const insightCards = [
  {
    tag: "Signals",
    title: "Why review volume matters",
    text: "Fake reviews often cluster in noisy bursts, which is why review shape matters as much as text content.",
  },
  {
    tag: "Pricing",
    title: "Discounts are not always savings",
    text: "A large markdown can be a clue. TrustLens compares the current price with the historical pattern.",
  },
  {
    tag: "Risk",
    title: "Seller trust is part of product trust",
    text: "Even a good product can be risky if the seller profile looks inconsistent, thin, or aggressive.",
  },
];

function formatRupee(value: number) {
  return new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatPercent(value: number) {
  return `${value.toFixed(1)}%`;
}

type ChartPoint = { x: number; y: number };

function buildSmoothPath(points: ChartPoint[]) {
  if (points.length < 2) {
    return "";
  }

  const commands = [`M ${points[0].x} ${points[0].y}`];

  for (let index = 0; index < points.length - 1; index += 1) {
    const current = points[index];
    const next = points[index + 1];
    const previous = points[index - 1] ?? current;
    const after = points[index + 2] ?? next;

    const control1X = current.x + (next.x - previous.x) / 6;
    const control1Y = current.y + (next.y - previous.y) / 6;
    const control2X = next.x - (after.x - current.x) / 6;
    const control2Y = next.y - (after.y - current.y) / 6;

    commands.push(`C ${control1X} ${control1Y}, ${control2X} ${control2Y}, ${next.x} ${next.y}`);
  }

  return commands.join(" ");
}

function buildSmoothAreaPath(points: ChartPoint[], baseline: number) {
  if (!points.length) {
    return "";
  }

  const linePath = buildSmoothPath(points);
  const closingPoints = `L ${points[points.length - 1].x} ${baseline} L ${points[0].x} ${baseline} Z`;
  return `${linePath} ${closingPoints}`;
}

function Gauge({ score, grade }: { score: number; grade: string }) {
  const radius = 74;
  const stroke = 10;
  const norm = radius - stroke / 2;
  const circumference = Math.PI * norm;
  const dash = (score / 100) * circumference;
  const color = score >= 85 ? "#10b981" : score >= 70 ? "#f59e0b" : "#ef4444";
  const verdict = score >= 85 ? "Trusted" : score >= 70 ? "Review carefully" : "High risk";

  return (
    <div className="gauge">
      <div className="gauge__eyebrow">Trust score</div>
      <svg width={radius * 2 + stroke} height={radius + stroke + 12} viewBox={`0 0 ${radius * 2 + stroke} ${radius + stroke + 12}`}>
        <path
          d={`M ${stroke / 2 + 2} ${radius + stroke / 2} A ${norm} ${norm} 0 0 1 ${radius * 2 + stroke - 2} ${radius + stroke / 2}`}
          fill="none"
          stroke="rgba(15, 23, 42, 0.12)"
          strokeWidth={stroke}
          strokeLinecap="round"
        />
        <path
          d={`M ${stroke / 2 + 2} ${radius + stroke / 2} A ${norm} ${norm} 0 0 1 ${radius * 2 + stroke - 2} ${radius + stroke / 2}`}
          fill="none"
          stroke={color}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={`${dash} ${circumference}`}
          style={{ filter: `drop-shadow(0 0 8px ${color}66)` }}
        />
        <text x={radius + stroke / 2} y={radius - 4} textAnchor="middle" fill={color} fontSize="34" fontWeight="800">
          {Math.round(score)}
        </text>
      </svg>
      <div className="gauge__value-row">
        <span className="gauge__score">/ 100</span>
        <span className="gauge__grade">Grade {grade}</span>
      </div>
      <div className="gauge__summary">
        <strong>{verdict}</strong>
        <span>{score >= 85 ? "Strong confidence zone" : score >= 70 ? "Mixed signals detected" : "Needs a closer look"}</span>
      </div>
      <div className="gauge__badge" style={{ color, borderColor: `${color}33`, background: `${color}14` }}>
        {verdict}
      </div>
    </div>
  );
}

function Sparkline({ history, isAnomaly }: { history: { month: string; price: number }[]; isAnomaly: boolean }) {
  const [activeIndex, setActiveIndex] = useState(Math.max(history.length - 1, 0));

  useEffect(() => {
    setActiveIndex(Math.max(history.length - 1, 0));
  }, [history.length]);

  if (!history.length) {
    return null;
  }

  const prices = history.map((entry) => entry.price);
  const min = Math.min(...prices);
  const max = Math.max(...prices);
  const spread = Math.max(max - min, 1);
  const width = 700;
  const height = 220;
  const paddingX = 34;
  const paddingY = 30;

  const x = (index: number) => paddingX + (index / Math.max(history.length - 1, 1)) * (width - paddingX * 2);
  const y = (value: number) => height - paddingY - ((value - min) / spread) * (height - paddingY * 2);
  const chartPoints = history.map((entry, index) => ({ x: x(index), y: y(entry.price) }));
  const linePath = buildSmoothPath(chartPoints);
  const areaPath = buildSmoothAreaPath(chartPoints, height - paddingY);
  const average = Math.round(prices.reduce((sum, price) => sum + price, 0) / prices.length);
  const active = history[Math.min(Math.max(activeIndex, 0), history.length - 1)];
  const latest = history[history.length - 1];
  const latestChange = history.length > 1 ? latest.price - history[history.length - 2].price : 0;
  const activeDelta = active.price - average;

  const setNearestPoint = (clientX: number, rectLeft: number) => {
    const ratio = (clientX - rectLeft - paddingX) / (width - paddingX * 2);
    const nextIndex = Math.round(ratio * Math.max(history.length - 1, 1));
    const bounded = Math.min(Math.max(nextIndex, 0), history.length - 1);
    setActiveIndex(bounded);
  };

  const setExactPoint = (index: number) => {
    if (index >= 0 && index < history.length) {
      setActiveIndex(index);
    }
  };

  return (
    <div className="sparkline-card">
      <div className="sparkline-card__header">
        <div>
          <span className="eyebrow">Price trend</span>
          <h3>Historical movement with exact price checkpoints</h3>
        </div>
        {isAnomaly ? (
          <span className="alert-pill">
            <TriangleAlert size={14} /> Anomaly detected
          </span>
        ) : (
          <span className="ok-pill">
            <CheckCircle2 size={14} /> Stable pattern
          </span>
        )}
      </div>
      <div className="sparkline__summary">
        <div>
          <span>Latest</span>
          <strong>{formatRupee(latest.price)}</strong>
          <small>{latestChange >= 0 ? "+" : "-"}{formatRupee(Math.abs(latestChange))} vs previous month</small>
        </div>
        <div>
          <span>Average</span>
          <strong>{formatRupee(average)}</strong>
          <small>Across {history.length} months</small>
        </div>
        <div>
          <span>Selected</span>
          <strong>{formatRupee(active.price)}</strong>
          <small>{active.month} checkpoint</small>
        </div>
      </div>
      <div className="sparkline__meta">
        <span className={`sparkline__chip ${active.price > average ? "sparkline__chip--up" : "sparkline__chip--down"}`}>
          {activeDelta >= 0 ? "+" : "-"}{formatRupee(Math.abs(activeDelta))} vs average
        </span>
        <span className="sparkline__note">Hover the line to inspect a month and compare it against the history.</span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="sparkline"
        preserveAspectRatio="none"
        onPointerMove={(event) => {
          const targetPoint = (event.target as Element | null)?.closest?.("[data-point-index]");
          const pointIndex = targetPoint ? Number(targetPoint.getAttribute("data-point-index")) : Number.NaN;

          if (!Number.isNaN(pointIndex)) {
            setExactPoint(pointIndex);
            return;
          }

          const rect = event.currentTarget.getBoundingClientRect();
          setNearestPoint(event.clientX, rect.left);
        }}
        onPointerLeave={() => setActiveIndex(history.length - 1)}
      >
        <defs>
          <linearGradient id="sparklineFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} stopOpacity="0.22" />
            <stop offset="75%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} stopOpacity="0.08" />
            <stop offset="100%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} stopOpacity="0" />
          </linearGradient>
          <linearGradient id="sparklineLine" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={isAnomaly ? "#d94646" : "#e3a11b"} />
            <stop offset="100%" stopColor={isAnomaly ? "#ef4444" : "#f59e0b"} />
          </linearGradient>
        </defs>
        <line x1={paddingX} x2={width - paddingX} y1={height - paddingY} y2={height - paddingY} stroke="rgba(15, 23, 42, 0.06)" strokeWidth="1.5" />
        <path d={areaPath} fill="url(#sparklineFill)" />
        <path d={linePath} fill="none" stroke="url(#sparklineLine)" strokeWidth="5" strokeLinecap="round" strokeLinejoin="round" />
        <line x1={x(activeIndex)} x2={x(activeIndex)} y1={paddingY - 6} y2={height - paddingY + 12} stroke="rgba(15, 23, 42, 0.14)" strokeDasharray="4 6" />
        {history.map((entry, index) => (
          <g key={`${entry.month}-${index}`} data-point-index={index}>
            <circle
              cx={x(index)}
              cy={y(entry.price)}
              r={16}
              fill="transparent"
              stroke="transparent"
            />
            <circle
              cx={x(index)}
              cy={y(entry.price)}
              r={index === activeIndex ? 9 : 6.5}
              fill={index === activeIndex ? (isAnomaly && index === history.length - 1 ? "#fff1f1" : "#fff7df") : "#fff"}
              stroke={isAnomaly && index === history.length - 1 ? "#ef4444" : index === activeIndex ? "#d97706" : "#f2b343"}
              strokeWidth={index === activeIndex ? 5 : 4}
              opacity={index === activeIndex ? 1 : 0.88}
            />
            <text x={x(index)} y={height - 4} textAnchor="middle" className="sparkline__label">
              {entry.month}
            </text>
          </g>
        ))}
      </svg>
      <div className="sparkline__tooltip">
        <div className="sparkline__tooltip-top">
          <strong>{active.month}</strong>
          <span>{formatRupee(active.price)}</span>
        </div>
        <small>
          {activeIndex === history.length - 1
            ? "Latest price checkpoint"
            : active.price > latest.price
              ? `${formatRupee(active.price - latest.price)} above latest`
              : `${formatRupee(latest.price - active.price)} below latest`}
        </small>
      </div>
    </div>
  );
}

function SectionHeading({ kicker, title, text }: { kicker: string; title: string; text: string }) {
  return (
    <div className="section-heading">
      <span className="eyebrow">{kicker}</span>
      <h2>{title}</h2>
      <p>{text}</p>
    </div>
  );
}

export default function App() {
  const [query, setQuery] = useState("https://amzn.in/d/0i8bjsw3");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState("");
  const [modelStats, setModelStats] = useState<ModelStats | null>(null);
  const [feedbackStatus, setFeedbackStatus] = useState("");
  const resultRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    fetch(`${API_BASE}/model-stats`)
      .then((response) => response.json())
      .then((data) => setModelStats(data))
      .catch(() => {
        setModelStats(null);
      });
  }, []);

  const analyze = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ url: trimmed }),
      });

      if (!response.ok) {
        const errorJson = await response.json().catch(() => null);
        throw new Error(errorJson?.detail || "Live product fetch failed. Please try again.");
      }

      const data: AnalysisResult = await response.json();
      setResult(data);

      window.setTimeout(() => {
        resultRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      }, 100);
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Analysis failed.";
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  const reportInaccuracy = async () => {
    if (!result) return;
    try {
      setFeedbackStatus("Submitting...");
      await fetch(`${API_BASE}/report-inaccuracy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url: query,
          title: result.title,
          reported_score: result.score,
          reported_risk: result.seller_risk.risk_level,
        }),
      });
      setFeedbackStatus("Thank you! Model retraining scheduled.");
    } catch {
      setFeedbackStatus("Failed to submit report.");
    }
  };

  const score = result?.score ?? 84;
  const grade = result?.grade ?? "A";
  const productTitle = result?.title ?? "Scan an Amazon product URL to see a live trust assessment";
  const productPrice = result?.price ?? 0;
  const productMrp = result?.mrp ?? 0;
  const previewHistory = result?.price_history ?? [
    { month: "Aug", price: 1299 },
    { month: "Sep", price: 1270 },
    { month: "Oct", price: 1315 },
    { month: "Nov", price: 1244 },
    { month: "Dec", price: 1238 },
    { month: "Jan", price: 1285 },
    { month: "Feb", price: 1248 },
    { month: "Mar", price: 1222 },
  ];

  const modelCards = Object.entries(modelStats?.models ?? {});

  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="container topbar__inner">
          <a href="#top" className="brand">
            <span className="brand__mark">
              <ShieldCheck size={18} />
            </span>
            <span>
              TrustLens <strong>AI</strong>
            </span>
          </a>

          <nav className="topbar__nav">
            <a href="#signals">Signals</a>
            <a href="#analyze">Analyzer</a>
            <a href="#stories">Stories</a>
            <a href="#insights">Insights</a>
          </nav>

          <a href="#analyze" className="btn btn--primary btn--small">
            Start scan
            <ArrowRight size={16} />
          </a>
        </div>
      </header>

      <main id="top">
        <section className="hero section">
          <div className="container hero__grid">
            <div className="hero__copy">
              <div className="eyebrow-row">
                <span className="eyebrow-chip">
                  <Sparkles size={14} /> Real Amazon trust scoring
                </span>
              </div>
              <h1>Know what’s safe before you buy.</h1>
              <p className="hero__lead">
                TrustLens AI analyzes Amazon product pages with a 5-model pipeline for fake reviews,
                price anomalies, sentiment quality, seller risk, and explainable trust scoring.
              </p>
              <div className="hero__actions">
                <a href="#analyze" className="btn btn--primary">
                  Analyze a product
                  <ArrowRight size={18} />
                </a>
                <a href="#signals" className="btn btn--secondary">
                  See how it works
                  <ChevronRight size={18} />
                </a>
              </div>
              <div className="hero__facts">
                {heroFacts.map((fact) => (
                  <span key={fact} className="fact-pill">
                    <BadgeCheck size={14} /> {fact}
                  </span>
                ))}
              </div>
            </div>

            <div className="hero__visual card card--hero">
              <div className="card__topline">
                <span className="eyebrow">Live analysis preview</span>
                <span className="status-chip">
                  <Clock3 size={14} /> {result?.analysis_time_s ? `${result.analysis_time_s}s scan` : "Instant preview"}
                </span>
              </div>

              <div className="visual-stack">
                <div className="visual-stack__backdrop" />
                <div className="visual-stack__panel">
                  <div className="visual-stack__panel-inner">
                    <div className="mock-product">
                      <div className="mock-product__image">
                        {result?.image ? (
                          <img src={result.image} alt={result.title} />
                        ) : (
                          <div className="mock-placeholder">
                            <Search size={32} />
                          </div>
                        )}
                      </div>
                      <div className="mock-product__body">
                        <div className="mock-product__meta">
                          <span>{result?.category ?? "Amazon product"}</span>
                          <span>{result?.brand ?? "Live scan"}</span>
                        </div>
                        <h3>{productTitle}</h3>
                        <div className="mock-product__pricing">
                          <strong>{productPrice ? formatRupee(productPrice) : "Awaiting URL"}</strong>
                          {productMrp > productPrice && <span>{formatRupee(productMrp)}</span>}
                        </div>
                        <div className="mock-product__chips">
                          <span>
                            <Store size={12} /> {result?.seller ?? "Seller profile"}
                          </span>
                          <span>
                            <ShieldCheck size={12} /> {result?.seller_risk?.risk_level ?? "Risk pending"}
                          </span>
                        </div>
                      </div>
                    </div>

                    <div className="visual-score-row">
                      <Gauge score={score} grade={grade} />
                      <div className="visual-score-row__copy">
                        <span className="eyebrow">Trust summary</span>
                        <h4>{result?.verdict ?? "A clean, premium summary of the product’s trust signals."}</h4>
                        <p>
                          {result?.source_mode === "live_scrape"
                            ? "Live Amazon data is active. The score is built from multiple trained models and transparent explanations."
                            : "Enter an Amazon product URL to pull a live result from the backend and inspect every signal."}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="mini-signal-grid">
                <div className="mini-signal-card">
                  <FileSearch size={18} />
                  <span>Review authenticity</span>
                </div>
                <div className="mini-signal-card">
                  <MessageCircleMore size={18} />
                  <span>Sentiment mismatch</span>
                </div>
                <div className="mini-signal-card">
                  <TrendingUp size={18} />
                  <span>Price anomaly</span>
                </div>
                <div className="mini-signal-card">
                  <BrainCircuit size={18} />
                  <span>Trust ensemble</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section className="section section--soft">
          <div className="container trust-strip">
            <div className="trust-strip__heading">
              <span className="eyebrow">Why people use it</span>
              <h2>Fast checks. Clear signals. Honest output.</h2>
            </div>
            <div className="trust-strip__grid">
              {storyCards.map((card) => (
                <article key={card.title} className="story-card card">
                  <h3>{card.title}</h3>
                  <p>{card.text}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="analyze" className="section">
          <div className="container analyzer-shell">
            <SectionHeading
              kicker="Analyzer"
              title="Scan any Amazon product URL"
              text="Paste a product link and get a clean, explainable trust assessment. No synthetic product output is shown if the backend cannot verify the page."
            />

            <form className="analysis-form card" onSubmit={analyze}>
              <label className="analysis-form__label" htmlFor="product-url">
                Amazon product URL
              </label>
              <div className="analysis-form__row">
                <input
                  id="product-url"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="https://amzn.in/d/... or a full amazon.in product page"
                />
                <button type="submit" className="btn btn--primary" disabled={loading}>
                  {loading ? "Analyzing..." : "Analyze now"}
                  {!loading && <ArrowRight size={18} />}
                </button>
              </div>
              <p className="analysis-form__note">
                Live-only output: if Amazon blocks the fetch, the backend returns a clear error instead of fake product data.
              </p>
            </form>

            {error && (
              <div className="error-banner">
                <TriangleAlert size={16} />
                <span>{error}</span>
              </div>
            )}

            {loading && (
              <div className="loading-grid">
                <div className="card skeleton-card" />
                <div className="card skeleton-card" />
                <div className="card skeleton-card" />
              </div>
            )}

            {!loading && !result && (
              <div className="analysis-empty card">
                <div>
                  <span className="eyebrow">What you’ll get</span>
                  <h3>Product summary, trust grade, and the reasons behind it.</h3>
                </div>
                <div className="analysis-empty__grid">
                  <div>
                    <strong>1</strong>
                    <span>Review authenticity</span>
                  </div>
                  <div>
                    <strong>2</strong>
                    <span>Seller and sentiment quality</span>
                  </div>
                  <div>
                    <strong>3</strong>
                    <span>Price and anomaly signals</span>
                  </div>
                </div>
              </div>
            )}

            {result && (
              <section ref={resultRef} className="result-area">
                <div className="result-top-grid">
                  <article className="card result-product">
                    <div className="result-product__image">
                      <img src={result.image} alt={result.title} />
                    </div>
                    <div className="result-product__body">
                      <div className="result-product__badges">
                        <span>{result.category}</span>
                        <span>{result.brand}</span>
                        <span className="badge-live">
                          <BadgeCheck size={12} /> Live scan
                        </span>
                      </div>
                      <h3>{result.title}</h3>
                      <div className="result-product__price-row">
                        <strong>{formatRupee(result.price)}</strong>
                        {result.mrp > result.price && <span>{formatRupee(result.mrp)}</span>}
                        {result.discount > 0 && <em>-{formatPercent(result.discount)}</em>}
                      </div>
                      <div className="result-product__meta-row">
                        <span>
                          <Store size={12} /> {result.seller}
                        </span>
                        <span>
                          <Star size={12} /> {result.rating} rating
                        </span>
                        <span>
                          <ShieldCheck size={12} /> {result.seller_risk.risk_level} risk
                        </span>
                      </div>
                    </div>
                  </article>

                  <article className="card result-score">
                    <span className="eyebrow">ML trust score</span>
                    <Gauge score={result.score} grade={result.grade} />
                    <div className="result-score__body">
                      <p>{result.verdict}</p>
                      <div className="result-score__stats">
                        <div>
                          <span>Score</span>
                          <strong>{Math.round(result.score)}/100</strong>
                        </div>
                        <div>
                          <span>Confidence</span>
                          <strong>{result.confidence_pct.toFixed(0)}%</strong>
                        </div>
                        <div>
                          <span>Risk label</span>
                          <strong>{result.seller_risk.risk_level}</strong>
                        </div>
                      </div>
                    </div>
                  </article>
                </div>

                <div className="result-detail-grid">
                  <article className="card detail-card">
                    <h4>
                      <FileSearch size={16} /> Review authenticity
                    </h4>
                    <div className="detail-kpi">
                      <strong>{result.fake_reviews.authenticity_score.toFixed(0)}%</strong>
                      <span>Authenticity score</span>
                    </div>
                    <div className="detail-list">
                      <div>
                        <span>Genuine reviews</span>
                        <strong>{result.fake_reviews.genuine_count}</strong>
                      </div>
                      <div>
                        <span>Suspicious reviews</span>
                        <strong>{result.fake_reviews.fake_count}</strong>
                      </div>
                      <div>
                        <span>Avg fake probability</span>
                        <strong>{formatPercent(result.fake_reviews.avg_fake_score * 100)}</strong>
                      </div>
                    </div>
                  </article>

                  <article className="card detail-card">
                    <h4>
                      <MessageCircleMore size={16} /> Sentiment analysis
                    </h4>
                    <div className="detail-kpi">
                      <strong>{result.sentiment.overall}</strong>
                      <span>Rating-language alignment</span>
                    </div>
                    <div className="detail-list">
                      <div>
                        <span>Positive</span>
                        <strong>{formatPercent(result.sentiment.sentiment_distribution.positive)}</strong>
                      </div>
                      <div>
                        <span>Neutral</span>
                        <strong>{formatPercent(result.sentiment.sentiment_distribution.neutral)}</strong>
                      </div>
                      <div>
                        <span>Negative</span>
                        <strong>{formatPercent(result.sentiment.sentiment_distribution.negative)}</strong>
                      </div>
                    </div>
                    {result.sentiment.mismatch_detected && (
                      <div className="warning-note">
                        <TriangleAlert size={16} />
                        <span>{result.sentiment.mismatch_reason}</span>
                      </div>
                    )}
                  </article>

                  <article className="card detail-card">
                    <h4>
                      <TrendingUp size={16} /> Price anomaly
                    </h4>
                    <div className="detail-kpi">
                      <strong>{result.price_anomaly.is_anomaly ? "Anomaly" : "Normal"}</strong>
                      <span>{formatPercent(result.price_anomaly.anomaly_score * 100)} anomaly score</span>
                    </div>
                    <div className="price-summary-grid">
                      <div>
                        <span>Current price</span>
                        <strong>{formatRupee(result.price)}</strong>
                      </div>
                      <div>
                        <span>MRP</span>
                        <strong>{formatRupee(result.mrp)}</strong>
                      </div>
                      <div>
                        <span>Savings</span>
                        <strong>{formatPercent(result.discount)}</strong>
                      </div>
                      <div>
                        <span>Vs history</span>
                        <strong>{formatPercent(result.price_anomaly.vs_avg_history)}</strong>
                      </div>
                    </div>
                    <div className="detail-list">
                      <div>
                        <span>Trend</span>
                        <strong>{result.price_anomaly.price_trend.replace(/_/g, " ")}</strong>
                      </div>
                      <div>
                        <span>Current vs MRP</span>
                        <strong>{result.mrp > result.price ? `${formatPercent(((result.mrp - result.price) / result.mrp) * 100)} off` : "No markdown"}</strong>
                      </div>
                      <div>
                        <span>Discount quality</span>
                        <strong>{formatPercent(result.price_anomaly.discount_pct)}</strong>
                      </div>
                    </div>
                  </article>
                </div>

                <div className="analysis-columns">
                  <article className="card signals-card">
                    <h4>
                      <Zap size={16} /> Positive signals
                    </h4>
                    <ul>
                      {result.pros.map((item) => (
                        <li key={item}>
                          <CheckCircle2 size={14} /> {item}
                        </li>
                      ))}
                    </ul>
                  </article>

                  <article className="card signals-card signals-card--warning">
                    <h4>
                      <TriangleAlert size={16} /> Risk signals
                    </h4>
                    <ul>
                      {result.cons.length > 0 ? (
                        result.cons.map((item) => (
                          <li key={item}>
                            <TriangleAlert size={14} /> {item}
                          </li>
                        ))
                      ) : (
                        <li>
                          <CheckCircle2 size={14} /> No major risk signals detected.
                        </li>
                      )}
                    </ul>
                  </article>
                </div>

                <Sparkline history={previewHistory} isAnomaly={result.price_anomaly.is_anomaly} />

                <div className="feedback-section" style={{ marginTop: '2rem', padding: '1.5rem', background: '#fff', borderRadius: '12px', border: '1px solid rgba(15, 23, 42, 0.08)', display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <div>
                    <h4 style={{ margin: 0, fontSize: '15px', fontWeight: 600 }}>Does this score look inaccurate?</h4>
                    <p style={{ margin: '4px 0 0', fontSize: '14px', color: '#64748b' }}>Your feedback helps retrain the AI models to improve accuracy.</p>
                  </div>
                  {feedbackStatus ? (
                    <span style={{ fontSize: '14px', fontWeight: 500, color: '#10b981' }}>{feedbackStatus}</span>
                  ) : (
                    <button onClick={reportInaccuracy} className="btn" style={{ background: '#f1f5f9', color: '#0f172a', display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <MessageSquareWarning size={16} /> Report inaccuracy
                    </button>
                  )}
                </div>
              </section>
            )}
          </div>
        </section>

        <section id="signals" className="section section--soft">
          <div className="container">
            <SectionHeading
              kicker="Signal map"
              title="Everything TrustLens checks"
              text="A quick overview of the model pipeline and the parts of the product page it evaluates before giving a trust score."
            />
            <div className="feature-grid">
              {featureCards.map((card) => {
                const Icon = card.icon;
                return (
                  <article key={card.title} className="card feature-card">
                    <div className="feature-card__icon">
                      <Icon size={20} />
                    </div>
                    <h3>{card.title}</h3>
                    <p>{card.text}</p>
                  </article>
                );
              })}
            </div>

            {modelCards.length > 0 && (
              <>
                <div className="section-heading section-heading--compact">
                  <span className="eyebrow">Model stats</span>
                  <h2>Backend models currently loaded</h2>
                </div>
                <div className="model-grid">
                  {modelCards.map(([key, model]) => (
                    <article key={key} className="card model-card">
                      <div className="model-card__top">
                        <div>
                          <span className="eyebrow">{key.replace(/_/g, " ")}</span>
                          <h3>{model.type}</h3>
                        </div>
                        <strong>
                          {model.accuracy ? formatPercent(model.accuracy * 100) : model.detection_rate ? formatPercent(model.detection_rate * 100) : "—"}
                        </strong>
                      </div>
                      <p>{model.algorithm}</p>
                      <small>{model.features}</small>
                    </article>
                  ))}
                </div>
              </>
            )}
          </div>
        </section>

        <section id="stories" className="section">
          <div className="container split-section">
            <div className="split-section__copy">
              <span className="eyebrow">Designed for clarity</span>
              <h2>Clean enough for shoppers, detailed enough for review work.</h2>
              <p>
                TrustLens is meant to feel premium and readable, with big type, warm spacing, and
                cards that do not fight each other for attention.
              </p>
              <div className="split-actions">
                <a href="#analyze" className="btn btn--primary">
                  Launch analyzer
                  <ArrowRight size={18} />
                </a>
                <a href="#insights" className="btn btn--secondary">
                  Explore insights
                  <ChevronRight size={18} />
                </a>
              </div>
            </div>

            <div className="split-section__visual card">
              <div className="mini-dashboard">
                <div>
                  <span className="eyebrow">Model summary</span>
                  <h3>{modelStats?.pipeline ?? "5-model ML pipeline with stacking ensemble"}</h3>
                </div>
                <div className="mini-dashboard__stats">
                  <div>
                    <strong>{modelStats?.total_models ?? 5}</strong>
                    <span>models</span>
                  </div>
                  <div>
                    <strong>{result?.score ?? 85}</strong>
                    <span>trust score</span>
                  </div>
                  <div>
                    <strong>{result?.analysis_time_s?.toFixed(2) ?? "2.40"}s</strong>
                    <span>analysis time</span>
                  </div>
                </div>
                <div className="mini-dashboard__checklist">
                  <div><BadgeCheck size={14} /> Live data</div>
                  <div><BadgeCheck size={14} /> Explainable result</div>
                  <div><BadgeCheck size={14} /> Model confidence</div>
                </div>
              </div>
            </div>
          </div>
        </section>

        <section id="insights" className="section section--light">
          <div className="container">
            <SectionHeading
              kicker="Latest insights"
              title="What makes a product look risky?"
              text="These short notes explain the signals TrustLens uses when building a verdict and highlight why the design focuses on transparency."
            />
            <div className="insight-grid">
              {insightCards.map((card) => (
                <article key={card.title} className="card insight-card">
                  <span className="insight-card__tag">{card.tag}</span>
                  <h3>{card.title}</h3>
                  <p>{card.text}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="section cta-band">
          <div className="container cta-band__inner">
            <div>
              <span className="eyebrow eyebrow--light">Ready to check a product?</span>
              <h2>Get a trust score before you buy.</h2>
              <p>
                Paste a live Amazon product URL and TrustLens will pull the data, score the risk,
                and show you why.
              </p>
            </div>
            <a href="#analyze" className="btn btn--primary btn--dark">
              Analyze now
              <ArrowRight size={18} />
            </a>
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="container footer__inner">
          <div>
            <a href="#top" className="brand brand--footer">
              <span className="brand__mark">
                <ShieldCheck size={18} />
              </span>
              <span>
                TrustLens <strong>AI</strong>
              </span>
            </a>
            <p>
              A premium Vite-based frontend for TrustLens AI, designed to feel lighter, sharper, and
              easier to trust.
            </p>
          </div>
          <div className="footer__meta">
            <span>
              <Layers3 size={14} /> 5-model pipeline
            </span>
            <span>
              <BarChart3 size={14} /> Live analysis dashboard
            </span>
            <span>
              <ShieldCheck size={14} /> Explainable verdicts
            </span>
          </div>
        </div>
      </footer>
    </div>
  );
}
