# Prediction-method-type-replacement

## Brief description
Python code for simulating infection disease dynamics of multi-type pathogens and the effect of vaccination. The code is used to compute the performance of predictors of type replacement proposed in "Capturing multiple-type interactions into practical predictors of type replacement following HPV vaccination", *Philosophical Transactions of the Royal Society of London. Series B, 2019 (doi: 10.1098/rstb.2018.0298)*.

### Prerequisites
- numpy
- scipy
- pandas
- csv
- pyDOE
- itertools
- time
- matplotlib
- seaborn

### Use
`parameters_generation.py`: This script generates random parameter sets used by `simulation.py`. 

`simulation.py`: This script simulates the pre- and post-vaccination steady states of multi-type dynamical systems, each corresponding to a generated parameter set. The default interaction structure (model) is `model = [mode = 'acq', multi = 'typewise', recip = 'structured', epsilon = '0']`.
It can also be changed to any of the following ten alternatives:
| mode     | multi       | recip          | epsilon |
| ---      | -----       | -----          | ------- |
| 'acq'    | 'groupwise' | 'structured'   | '0'     |
| 'acq'    | 'typewise'  | 'structured'   | '0'     |
| 'acq'    | 'typewise'  | 'structured'   | '2'     |
| 'acq'    | 'typewise'  | 'structured'   | '5'     |
| 'acq'    | 'typewise'  | 'unstructured' | '0'     |
| 'acq_cl' | 'groupwise' | 'structured'   | '0'     |
| 'acq_cl' | 'typewise'  | 'structured'   | '0'     |
| 'acq_cl' | 'typewise'  | 'structured'   | '2'     |
| 'acq_cl' | 'typewise'  | 'structured'   | '5'     |
| 'acq_cl' | 'typewise'  | 'unstructured' | '0'     |

`performance.py`: Using the pre- and post-vaccination steady states simulated by simulation.py, this script computes the values of the proposed predictors, validates the predicted outcomes against the true outcomes of vaccination, and so compute the performance of the predictors. It produces heatmaps of performance for different combinations of vaccine and non-vaccine types. Note that the interaction structure (model) needs to set identically to the one used in `simulation.py`. Below is the performance with `model = [mode = 'acq', multi = 'typewise', recip = 'structured', epsilon = '0']`.

![exemple](main_performance_HR_VTi.png?raw=true)

### Authors
<table>
  <tr>
    <td>Irene Man</th>
    <td>National Institute for Public Health and the Environment</th>
    <td>The Netherlands</th>
  </tr>
  <tr>
    <td>Kari Auranen</td>
    <td>Turku University</td>
    <td>Finland</td>
  </tr>
  <tr>
    <td>Jacco Wallinga</th>
    <td>National Institute for Public Health and the Environment</th>
    <td>The Netherlands</th>
  </tr>
  <tr>
    <td>Johannes A. Bogaards</th>
    <td>National Institute for Public Health and the Environment</th>
    <td>The Netherlands</th>
  </tr>
</table>
