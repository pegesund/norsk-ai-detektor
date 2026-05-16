//! Norsk AI-detektor — Yew GUI for menneske/AI-klassifisering.
//!
//! Inference + tokenisering gjøres via transformers.js (eksponert via window.naiDetector).
//! Modellen lastes fra HuggingFace ved første scoring.

use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use yew::prelude::*;

// Lengde-grenser — basert på trening-distribusjon. Under ~50 tegn er signalet svakt.
const MIN_CHARS: usize = 50;
const RECOMMENDED_MIN: usize = 250;
const RECOMMENDED_MAX: usize = 12_000; // ~2000 ord; over dette tokeniseres bare første 1024 tokens
const HARD_MAX: usize = 50_000;

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = ["window", "naiDetector"], js_name = loadModel)]
    fn js_load_model(progress_cb: &JsValue) -> js_sys::Promise;

    #[wasm_bindgen(js_namespace = ["window", "naiDetector"], js_name = predict)]
    fn js_predict(text: &str) -> js_sys::Promise;
}

#[derive(Clone, PartialEq, Debug, Default)]
struct LoadProgress {
    file: String,
    percent: f64,
    loaded_mb: f64,
    total_mb: f64,
}

#[derive(Clone, PartialEq, Debug)]
enum ModelState {
    NotLoaded,
    Loading(LoadProgress),
    Loaded,
    Error(String),
}

#[derive(Clone, PartialEq, Debug)]
struct Prediction {
    p_human: f64,
    p_ai: f64,
    cleaned_chars: usize,
}

#[function_component(App)]
fn app() -> Html {
    let text = use_state(|| String::new());
    let model_state = use_state(|| ModelState::NotLoaded);
    let prediction = use_state(|| None::<Prediction>);
    let busy = use_state(|| false);

    let on_text_input = {
        let text = text.clone();
        Callback::from(move |e: InputEvent| {
            let target: web_sys::HtmlTextAreaElement = e.target_unchecked_into();
            text.set(target.value());
        })
    };

    let on_score = {
        let text = text.clone();
        let model_state = model_state.clone();
        let prediction = prediction.clone();
        let busy = busy.clone();
        Callback::from(move |_| {
            let t = (*text).clone();
            if t.chars().count() < MIN_CHARS {
                return;
            }
            let model_state = model_state.clone();
            let prediction = prediction.clone();
            let busy = busy.clone();
            wasm_bindgen_futures::spawn_local(async move {
                busy.set(true);

                // Last modell hvis ikke lastet
                if matches!(*model_state, ModelState::NotLoaded) {
                    model_state.set(ModelState::Loading(LoadProgress::default()));
                    // Progress-callback fra transformers.js
                    let ms = model_state.clone();
                    let cb = Closure::wrap(Box::new(move |info: JsValue| {
                        gloo::console::log!("[rust progress]", &info);
                        let status = js_sys::Reflect::get(&info, &"status".into())
                            .ok().and_then(|v| v.as_string()).unwrap_or_default();
                        let file = js_sys::Reflect::get(&info, &"file".into())
                            .ok().and_then(|v| v.as_string()).unwrap_or_default();
                        let name = js_sys::Reflect::get(&info, &"name".into())
                            .ok().and_then(|v| v.as_string()).unwrap_or_default();
                        let progress = js_sys::Reflect::get(&info, &"progress".into())
                            .ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let loaded = js_sys::Reflect::get(&info, &"loaded".into())
                            .ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
                        let total = js_sys::Reflect::get(&info, &"total".into())
                            .ok().and_then(|v| v.as_f64()).unwrap_or(0.0);
                        // Vis noe for ALLE events (ikke filtrer) — bruk file/name som fallback
                        let label = if !file.is_empty() { file }
                                    else if !name.is_empty() { name }
                                    else { status.clone() };
                        ms.set(ModelState::Loading(LoadProgress {
                            file: label,
                            percent: progress.min(100.0),
                            loaded_mb: loaded / (1024.0 * 1024.0),
                            total_mb: total / (1024.0 * 1024.0),
                        }));
                    }) as Box<dyn FnMut(JsValue)>);
                    let res = JsFuture::from(js_load_model(cb.as_ref())).await;
                    cb.forget(); // lever ut levetiden til Rust
                    match res {
                        Ok(_) => model_state.set(ModelState::Loaded),
                        Err(e) => {
                            let msg = format!("{:?}", e);
                            model_state.set(ModelState::Error(msg));
                            busy.set(false);
                            return;
                        }
                    }
                }

                // Predict
                match JsFuture::from(js_predict(&t)).await {
                    Ok(val) => {
                        let p_human = js_sys::Reflect::get(&val, &"p_human".into())
                            .ok()
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let p_ai = js_sys::Reflect::get(&val, &"p_ai".into())
                            .ok()
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0);
                        let cleaned_chars = js_sys::Reflect::get(&val, &"cleaned_chars".into())
                            .ok()
                            .and_then(|v| v.as_f64())
                            .unwrap_or(0.0) as usize;
                        prediction.set(Some(Prediction {
                            p_human,
                            p_ai,
                            cleaned_chars,
                        }));
                    }
                    Err(e) => {
                        gloo::console::error!("predict feilet:", &e);
                    }
                }
                busy.set(false);
            });
        })
    };

    let on_clear = {
        let text = text.clone();
        let prediction = prediction.clone();
        Callback::from(move |_| {
            text.set(String::new());
            prediction.set(None);
        })
    };

    let n = text.chars().count();
    let length_class = if n == 0 {
        "length-info"
    } else if n < MIN_CHARS {
        "length-info err"
    } else if n < RECOMMENDED_MIN || n > RECOMMENDED_MAX {
        "length-info warn"
    } else {
        "length-info"
    };

    let length_warning = if n > 0 && n < MIN_CHARS {
        Some(html! {
            <div class="warning-box err">
                <strong>{"For kort."}</strong>{" Skriv inn minst "}{MIN_CHARS}{" tegn. Modellen trenger nok tekst for å gi et meningsfullt svar."}
            </div>
        })
    } else if n > 0 && n < RECOMMENDED_MIN {
        Some(html! {
            <div class="warning-box">
                <strong>{"Kort tekst."}</strong>{" Modellen er mest pålitelig på tekster over ~"}{RECOMMENDED_MIN}{" tegn (~50 ord)."}
            </div>
        })
    } else if n > HARD_MAX {
        Some(html! {
            <div class="warning-box err">
                <strong>{"For lang."}</strong>{" Maks "}{HARD_MAX}{" tegn."}
            </div>
        })
    } else if n > RECOMMENDED_MAX {
        Some(html! {
            <div class="warning-box">
                <strong>{"Lang tekst."}</strong>{" Modellen ser bare første ~1024 tokens (ca. 6 000 tegn). Resten av teksten påvirker ikke vurderingen."}
            </div>
        })
    } else {
        None
    };

    let status_text = match &*model_state {
        ModelState::NotLoaded => html! { <></> },
        ModelState::Loading(p) => {
            let file_label = if p.file.is_empty() { "modellen".to_string() } else { p.file.clone() };
            let label = if p.total_mb > 0.0 {
                format!("Laster {} — {:.0}% ({:.1} / {:.1} MB)",
                        file_label, p.percent, p.loaded_mb, p.total_mb)
            } else if !p.file.is_empty() {
                format!("Henter {} ...", file_label)
            } else {
                "Forbereder lasting...".to_string()
            };
            // Hvis vi ikke har bytes ennå, vis indeterminate progress (uten value)
            let progress_html = if p.total_mb > 0.0 {
                let pct = p.percent / 100.0;
                html! { <progress value={format!("{}", pct)} max="1" /> }
            } else {
                html! { <progress /> }
            };
            html! {
                <div class="status">
                    <div style="margin-bottom: 0.5rem;">{ label }</div>
                    { progress_html }
                </div>
            }
        }
        ModelState::Loaded => html! { <></> },
        ModelState::Error(e) => html! {
            <div class="status" style="color: var(--pico-color-red-550);">
                {"Feil: "}{e.clone()}
            </div>
        },
    };

    let result = prediction.as_ref().map(|p| {
        let is_human = p.p_human > p.p_ai;
        let verdict_class = if is_human { "result human" } else { "result ai" };
        let verdict_text = if is_human { "Sannsynligvis menneske" } else { "Sannsynligvis AI-generert" };
        let confidence = (p.p_human.max(p.p_ai) * 100.0).round();
        html! {
            <div class={verdict_class}>
                <div class="verdict">{verdict_text}</div>
                <div>{format!("Konfidens: {}%", confidence)}</div>
                <div class="bars">
                    <div>
                        <div class="bar-label">
                            <span>{"P(menneske)"}</span>
                            <span>{format!("{:.2}%", p.p_human * 100.0)}</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill human" style={format!("width: {}%", p.p_human * 100.0)} />
                        </div>
                    </div>
                    <div>
                        <div class="bar-label">
                            <span>{"P(AI)"}</span>
                            <span>{format!("{:.2}%", p.p_ai * 100.0)}</span>
                        </div>
                        <div class="bar">
                            <div class="bar-fill ai" style={format!("width: {}%", p.p_ai * 100.0)} />
                        </div>
                    </div>
                </div>
                <p style="font-size: 0.8rem; color: var(--pico-muted-color); margin-top: 1rem;">
                    {format!("Tekst etter normalisering: {} tegn", p.cleaned_chars)}
                </p>
            </div>
        }
    });

    let score_disabled = *busy || n < MIN_CHARS || n > HARD_MAX;

    html! {
        <main class="container">
            <header>
                <h1>{"Norsk AI-detektor"}</h1>
                <p class="lead">
                    {"Vurder om en tekst er skrevet av et menneske eller generert av AI. "}
                    {"Modellen kjører lokalt i nettleseren din — teksten din forlater aldri maskinen."}
                </p>
                <details class="intro-box">
                    <summary>{"Slik fungerer det · trent på · førstegangslasting"}</summary>
                    <p>
                        <strong>{"Førstegangslasting tar tid."}</strong>{" Når du klikker «Analyser tekst» "}
                        {"første gang, lastes hele modellen (~90 MB) ned fra HuggingFace. Det kan ta "}
                        {"1-3 minutter avhengig av forbindelsen din. Modellen blir så cachet i nettleseren — "}
                        {"alle senere analyser går på under et sekund."}
                    </p>
                    <p>
                        <strong>{"AI-modeller den er trent til å gjenkjenne:"}</strong>{" Anthropic Claude (Opus, Sonnet, Haiku 4.x), "}
                        {"OpenAI GPT-5 og GPT-5-mini, Google Gemini 2.5 Pro og Flash. "}
                        {"Tekst fra modeller utenfor denne listen er ikke testet."}
                    </p>
                    <p>
                        <strong>{"Åpen og fri."}</strong>{" Selve modellen ligger åpent tilgjengelig på "}
                        <a href="https://huggingface.co/pegesund/norwegian-ai-detector" target="_blank" rel="noopener">{"HuggingFace"}</a>
                        {" under CC-BY-SA-4.0. Kildekoden til denne websiden ligger på "}
                        <a href="https://github.com/pegesund/norsk-ai-detektor" target="_blank" rel="noopener">{"GitHub"}</a>{"."}
                    </p>
                </details>
            </header>

            <div class="textarea-wrap">
                <textarea
                    placeholder="Lim inn norsk tekst her (bokmål eller nynorsk)..."
                    value={(*text).clone()}
                    oninput={on_text_input}
                />
                <div class={length_class}>
                    <span>{format!("{} tegn", n)}</span>
                    <span>{"Anbefalt: 250–12 000 tegn"}</span>
                </div>
            </div>

            { length_warning.unwrap_or_default() }

            <div class="btn-row">
                <button onclick={on_score} disabled={score_disabled}>
                    if *busy {
                        {"Analyserer..."}
                    } else {
                        {"Analyser tekst"}
                    }
                </button>
                <button class="secondary outline" onclick={on_clear}>
                    {"Tøm"}
                </button>
            </div>

            { status_text }
            { for result }

            <footer>
                <p>
                    {"Modell: "}
                    <a href="https://huggingface.co/pegesund/norwegian-ai-detector" target="_blank" rel="noopener">
                        {"pegesund/norwegian-ai-detector"}
                    </a>
                    {" · Baseline: "}
                    <a href="https://huggingface.co/ltg/norbert4-small" target="_blank" rel="noopener">
                        {"NorBERT-4-small"}
                    </a>
                </p>
                <p>
                    {"In-distribution F1 = 0.998. Reell ytelse på ukjente AI-modeller og tekster utenfor treningssettet er ikke garantert."}
                </p>
            </footer>
        </main>
    }
}

fn main() {
    yew::Renderer::<App>::new().render();
}
