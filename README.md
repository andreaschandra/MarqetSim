# MarqetSim

MarqetSim is the user interface (UI) component for TinyTroupe, providing a seamless and intuitive experience for users interacting with the TinyTroupe platform.

## Overview

MarqetSim serves as the front-end application for TinyTroupe, offering a visual and interactive interface for users to engage with the platform's features and functionalities.

## Features

- UI for Run Simulation
- CLI capability

## Getting Started for UI

### Prerequisites

- depenencies are in requirements.xtt

### Installation

1. Clone the repository:

```
git clone git@github.com>:andreaschandra/MarqetSim.git
```

2. Navigate to the project directory:

```
cd MarQetSim
```

3. Install dependencies:

```
pip install -r requirements.txt
```

### Running the Application

To start the development server:

```
python app.py
```

It will show something like

```
Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```

## Getting Started for CLI

run this (using uv framework)

```
uv pip install -e .
```

run sample experiment

```
marq launch test/projects_config/market-insights-ai-num-agents.yaml

```

check summarize results
```
marq summarize test-results/config-name/test.csv
```
