// Generator for the NVIDIA Academic Grant proposal .docx (Domenico's revised text, 2026-06).
// Run: node build_proposal.js  ->  nvidia_academic_grant_2026_template.docx
const fs=require('fs');
const {Document,Packer,Paragraph,TextRun,Table,TableRow,TableCell,HeadingLevel,
       AlignmentType,BorderStyle,WidthType,ShadingType,LevelFormat,PageBreak}=require('docx');

const PW=12240, M=1440, CW=PW-2*M;
const border={style:BorderStyle.SINGLE,size:1,color:"BBBBBB"};
const borders={top:border,bottom:border,left:border,right:border};
const cellM={top:80,bottom:80,left:120,right:120};

function H1(t){return new Paragraph({heading:HeadingLevel.HEADING_1,children:[new TextRun(t)]});}
function P(runs,opts={}){return new Paragraph(Object.assign({children:(Array.isArray(runs)?runs:[new TextRun(runs)])},opts));}
function plain(t){return P([new TextRun(t)]);}
function sub(t){return P([new TextRun({text:t,bold:true})],{spacing:{before:120,after:40}});}
function bold(label,rest){return P([new TextRun({text:label,bold:true}),new TextRun(rest||"")]);}
function bullet(t){return new Paragraph({numbering:{reference:"b",level:0},children:[new TextRun(t)]});}
function cell(text,{head=false,w}={}){return new TableCell({borders,width:{size:w,type:WidthType.DXA},margins:cellM,
  shading: head?{fill:"E8EEF5",type:ShadingType.CLEAR}:undefined,
  children:[P([new TextRun({text,bold:head})])]});}

const c1=1700,c2=5760,c3=1900;
const timeline=new Table({width:{size:CW,type:WidthType.DXA},columnWidths:[c1,c2,c3],rows:[
  new TableRow({children:[cell("Month",{head:true,w:c1}),cell("Activity",{head:true,w:c2}),cell("A100-hours",{head:true,w:c3})]}),
  new TableRow({children:[cell("Oct 2026",{w:c1}),cell("Cloud onboarding; calibrate workload intensity per platform on A100 (Triton, Dynamo, vLLM; Nemotron and Qwen).",{w:c2}),cell("~200",{w:c3})]}),
  new TableRow({children:[cell("Nov 2026",{w:c1}),cell("A100 base factorial, first half; per-run reliability analysis.",{w:c2}),cell("~600",{w:c3})]}),
  new TableRow({children:[cell("Dec 2026",{w:c1}),cell("A100 base factorial, second half; consolidate platform x hardware x model comparison.",{w:c2}),cell("~600",{w:c3})]}),
  new TableRow({children:[cell("Jan 2027",{w:c1}),cell("7-day exploratory runs on non-converged configurations; NVIDIA Nsight profiling.",{w:c2}),cell("~700",{w:c3})]}),
  new TableRow({children:[cell("Feb 2027",{w:c1}),cell("Stress-workload probe: vary request size, repetition, and burstiness.",{w:c2}),cell("~350",{w:c3})]}),
  new TableRow({children:[cell("Mar 2027",{w:c1}),cell("Conditional 14-day confirmation experiments; data egress.",{w:c2}),cell("~700",{w:c3})]}),
  new TableRow({children:[cell("Apr 2027",{w:c1}),cell("Analysis, write-up, open-source release; reserve for re-runs.",{w:c2}),cell("~100",{w:c3})]}),
]});

const children=[
  P([new TextRun({text:"NVIDIA Academic Grant Program - Research Proposal",bold:true,size:28})],{alignment:AlignmentType.CENTER,spacing:{after:80}}),
  P([new TextRun({text:"Interest Area: AI Inference, Agents, and Systems Software",italics:true})],{alignment:AlignmentType.CENTER,spacing:{after:140}}),
  P([new TextRun({text:"Understanding Silent Capacity Erosion and Long-Term Reliability in NVIDIA AI Inference Infrastructure",bold:true,size:26})],{alignment:AlignmentType.CENTER,spacing:{after:140}}),
  bold("Principal Investigator: ","Domenico Cotroneo, Professor, UNC Charlotte"),

  H1("Abstract"),
  plain("AI inference is rapidly becoming critical infrastructure: modern LLM serving platforms operate continuously on expensive GPUs and are expected to deliver stable performance for weeks or months. Yet, despite extensive work on throughput and latency, little is known about their long-term reliability. Our preliminary experiments suggest that GPU-based LLM serving stacks may silently accumulate resources over time while client-visible performance stays stable, so a deployment can progressively lose serving capacity, with no visible warning, increasing its exposure to resource-exhaustion conditions and service disruption. These are previously unexplored reliability risks, with implications for service availability, capacity planning, and operational cost."),
  plain("This project extends that preliminary evidence across NVIDIA serving platforms and hardware (NVIDIA Triton Inference Server, NVIDIA Dynamo, and A100 GPUs). Through repeated multi-day experiments, realistic workloads, and synchronized host, process, GPU, and client monitoring, we will test whether these reliability signatures generalize, identify the software components responsible for long-term resource accumulation, and assess whether workload characteristics can accelerate degradation. The result will be the first systematic understanding of long-term reliability risks in NVIDIA AI inference infrastructure, with open-source tools and practical guidance for building more dependable AI services."),
  bold("Project Keywords: ","long-term reliability, AI infrastructure, LLM serving, NVIDIA Triton, NVIDIA Dynamo, resource accumulation, silent capacity erosion"),

  H1("NVIDIA Platforms"),
  bullet("Software: NVIDIA Triton Inference Server and NVIDIA Dynamo are the inference platforms under study. NVIDIA NGC containers will provide reproducible deployment environments across all experiments."),
  bullet("Models: Two served models will be used throughout the factorial design: a pretrained NVIDIA Nemotron model and Qwen2.5-7B, enabling comparison between an NVIDIA-native model and a widely adopted open-source model."),
  bullet("Profiling: NVIDIA Nsight Systems and NVIDIA Nsight Compute will support root-cause analysis of resource accumulation phenomena and memory-management behavior."),
  bullet("Hardware: NVIDIA A100 GPUs will provide the cloud environment for long-duration reliability experiments and cross-platform comparisons."),

  H1("Dataset and Models"),
  plain("We do not train models; the primary data generated by this project consists of workload traces and long-duration monitoring measurements."),
  bullet("Workload: An open-loop Poisson client will generate requests from an arXiv-based prompt corpus (~3,000 prompts) with log-normal prompt and output-length distributions. No personal, proprietary, or confidential data will be used."),
  bullet("Served Models: a pretrained NVIDIA Nemotron model and Qwen2.5-7B-Instruct, used throughout the factorial design (continuity with our preliminary study)."),
  bullet("Outputs: The project will generate approximately 1 TB of monitoring traces spanning process-level, GPU-level, system-level, and client-side measurements. A curated subset will be released as an open research dataset."),

  H1("Introduction"),
  plain("Our preliminary study [1] revealed a concerning phenomenon in GPU-based LLM serving: while client-side indicators such as latency, throughput, and error rates stayed stable, internal resources kept accumulating over time. The deployments looked healthy from the user side while quietly consuming more resources in the background."),
  plain("This observation raises an important reliability concern. If the accumulation stays invisible to conventional operational metrics, a deployment can suffer a silent erosion of serving capacity long before operators notice, reducing resilience to workload surges, increasing susceptibility to resource-exhaustion conditions, and ultimately threatening availability in large-scale AI deployments. That study was intentionally exploratory and limited to a single hardware platform, a single model, and short observation windows. Whether the behavior generalizes across NVIDIA inference platforms, GPU architectures, and deployment configurations remains open, which motivates this project."),
  plain("This project addresses the following research questions:"),
  bold("RQ1. Generalization: ","Do long-term reliability signatures generalize across NVIDIA inference platforms, GPU architectures, and served models?"),
  bold("RQ2. Mechanisms: ","Which software components are responsible for resource accumulation and degradation in modern LLM serving stacks?"),
  bold("RQ3. Operational Impact: ","Can workload characteristics accelerate resource accumulation to the point where a seemingly healthy deployment experiences a silent loss of serving capacity and increased susceptibility to service disruption?"),

  H1("Methods"),
  plain("To address RQ1, we compare NVIDIA Triton, NVIDIA Dynamo, and standalone vLLM across NVIDIA A100 and L40S GPUs using repeated multi-day experiments under controlled workloads. To address RQ2, we combine system-level monitoring with NVIDIA Nsight Systems and Nsight Compute to identify the software layers responsible for resource accumulation. To address RQ3, we vary workload characteristics (request size, repetition, and burstiness) to assess whether accumulation can reduce serving capacity and increase susceptibility to resource exhaustion and service disruption."),
  plain("All experiments run in reproducible NVIDIA NGC containers and collect synchronized host, process, GPU, and client measurements. Statistical analysis follows our preliminary study and separates genuine long-term trends from normal workload fluctuations."),
  bold("Workload Calibration. ","Before each long-duration experiment, we determine the throughput ceiling of the target deployment and configure the workload as a fixed fraction of that ceiling. This normalization ensures comparable stress levels across platforms and GPU architectures."),
  bold("Experimental Design. ","We adopt a full-factorial Design of Experiments (DOE)."),
  sub("Factors"),
  bullet("Inference Platform: NVIDIA Triton, NVIDIA Dynamo, standalone vLLM"),
  bullet("GPU Platform: NVIDIA L40S, NVIDIA A100"),
  bullet("Served Model: NVIDIA Nemotron, Qwen2.5-7B"),
  plain("Each treatment combination is replicated three times to estimate main and interaction effects. Configurations exhibiting statistically significant resource accumulation or silent capacity erosion will be subjected to extended-duration experiments (up to seven days) and targeted workload-stress campaigns."),

  H1("Proposed Timeline"),
  timeline,

  H1("Expected Outcomes and Impact"),
  plain("The project will produce the first systematic characterization of long-term reliability risks in NVIDIA-based AI inference infrastructure, establishing long-term reliability as an evaluation dimension alongside throughput and latency."),
  plain("Concrete outcomes include:"),
  bullet("An open-source measurement and analysis framework and a curated dataset of monitoring traces across platforms, models, and GPU configurations."),
  bullet("Identification of the software components and deployment configurations associated with resource accumulation and degradation."),
  bullet("Characterization of workload conditions that amplify reliability risks, with practical guidance for detecting and mitigating silent capacity erosion in production."),
  bullet("Publications targeting the dependability and AI-systems communities."),
  plain("If reliability issues are found in specific software components, results will be responsibly disclosed to the relevant open-source and industrial stakeholders before publication."),

  H1("Project Support Details"),
  plain("The proposed experimental campaign is based on long-duration measurements rather than compute-intensive model training. As a result, GPU consumption is primarily determined by experiment duration and concurrency rather than sustained peak utilization. The requested resources are sized to support the full-factorial experimental design, including independent repetitions, extended-duration investigations, and confirmation runs."),
  plain("The project team has already developed and validated the complete experimental workflow on local NVIDIA L40S infrastructure, including workload generation, monitoring, data collection, statistical analysis, containerized deployment, and automation. The requested NVIDIA A100 resources will therefore be used to extend an existing and operational experimental framework to additional hardware configurations and longer observation windows."),
  bold("Cloud Hours (A100): ","5,000"),
  bold("Concurrent GPUs: ","up to 6"),
  bold("Cloud Storage: ","approximately 1 TB for monitoring traces, logs, and containerized environments"),
  plain("GPU-hour breakdown. Base A100 = 18 runs x 48h: vLLM 6x48=288, Triton 6x48=288, Dynamo (2 GPU) 6x48x2=576, for ~1,152 hours. Extended-duration runs ~672, stress-workload probe ~300, conditional 14-day confirmation ~672, and calibration ~150, totaling ~2,950 hours of planned compute (about 3,250 across the schedule). The requested 5,000 hours include margin for re-runs and preemptions. The L40S half of the factorial runs locally at no cost to the grant."),

  H1("Cloud Readiness"),
  plain("The complete experimental workflow is already operational on local NVIDIA L40S infrastructure (Ubuntu 22.04) and migrates directly to cloud NVIDIA A100. Our first activities are routine for the team: pull the prompt corpus and code from version-controlled repositories; install pinned dependencies; launch the bash-driven campaign and analysis pipeline already validated locally; sync traces and logs to persistent object storage; and run the NVIDIA stack (Triton, Dynamo, CUDA, Nsight) through pinned NGC and Docker containers for reproducibility. Experience with all of these is high."),

  H1("Investigators"),
  plain("Domenico Cotroneo is a Professor at UNC Charlotte with more than two decades of research experience in software dependability, software aging, and performance evaluation. His work has contributed to the development of measurement-based methodologies for detecting and characterizing long-term degradation phenomena in complex software systems [3, 5], including operating systems, managed runtimes [2], mobile platforms [6], virtualized environments, and distributed infrastructures."),
  plain("His research has consistently followed the evolution of computing infrastructures, extending software aging and reliability assessment methodologies from traditional systems to edge-AI platforms [4] and, more recently, GPU-based LLM serving infrastructures. This proposal builds directly on a recent preliminary study by his group [1] that provided early evidence of resource accumulation phenomena in modern LLM serving systems."),
  plain("His expertise combines experimental design, statistical analysis of long-running systems, and dependability assessment of emerging computing infrastructures. These capabilities provide the methodological foundation required to investigate long-term reliability risks in NVIDIA AI inference platforms and to translate experimental findings into practical guidance for building more dependable AI services."),

  H1("References"),
  plain("[1] D. Cotroneo and B. Cukic, Characterizing Software Aging in GPU-Based LLM Serving Systems, arXiv:2606.11916, 2026."),
  plain("[2] D. Cotroneo, S. Orlando, R. Pietrantuono, and S. Russo, “A measurement-based aging analysis of the JVM,” Software Testing, Verification and Reliability, 2013."),
  plain("[3] D. Cotroneo, R. Natella, R. Pietrantuono, and S. Russo, “Software aging and rejuvenation: where we are and where we are going,” WoSAR, 2011."),
  plain("[4] K. Watanabe et al., “Software aging in a real-time object detection system on an edge server,” SAC, 2023."),
  plain("[5] C. Zhang et al., “SGT: Aging-related bug prediction via semantic feature learning based on graph-transformer,” Journal of Systems and Software, 2024."),
  plain("[6] D. Cotroneo et al., “Software micro-rejuvenation for Android mobile systems,” Journal of Systems and Software, 2022."),
];

const doc=new Document({
  creator:"Domenico Cotroneo",
  lastModifiedBy:"Domenico Cotroneo",
  title:"NVIDIA Academic Grant Proposal - Long-Term Reliability of AI Inference",
  description:"Research proposal",
  keywords:"long-term reliability; AI infrastructure; LLM serving; NVIDIA Triton; NVIDIA Dynamo",
  styles:{default:{document:{run:{font:"Arial",size:22}}},
    paragraphStyles:[
      {id:"Heading1",name:"Heading 1",basedOn:"Normal",next:"Normal",quickFormat:true,
        run:{size:26,bold:true,font:"Arial",color:"1F3864"},
        paragraph:{spacing:{before:220,after:100},outlineLevel:0}}]},
  numbering:{config:[{reference:"b",levels:[{level:0,format:LevelFormat.BULLET,text:"•",alignment:AlignmentType.LEFT,
    style:{paragraph:{indent:{left:540,hanging:270}}}}]}]},
  sections:[{properties:{page:{size:{width:12240,height:15840},margin:{top:M,right:M,bottom:M,left:M}}},children}]
});
Packer.toBuffer(doc).then(b=>{fs.writeFileSync("nvidia_academic_grant_2026_template.docx",b);console.log("written",b.length,"bytes");});
