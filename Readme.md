This is the code to replicate the analysis in the paper
" Zhang, G., Cai, B., Zhang, A., Tu, Z., Xiao, L., Stephen, J. M., ... & Wang, Y. P. (2022). Detecting abnormal connectivity in schizophrenia via a joint directed acyclic graph estimation model. Neuroimage, 260, 119451. "
https://www.sciencedirect.com/science/article/pii/S1053811922005687

utils_joint.py is the main source file for all functions and tools

joint_notears.py is the main function to implement the joint directed networks estimation with an example using synthetic data.

All synthetic data are generated using generate_simulation.py.

notears_model_selection.py is the code used for model selection regarding the two paramters, i.e., $\lambda_1$ and $\lambda_2$.
