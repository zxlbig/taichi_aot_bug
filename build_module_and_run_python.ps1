$tiModuleSdfCommand = "python -m taichi._main module build sdf.py -o .\assets\sdf.tcm"
Invoke-Expression -Command $tiModuleSdfCommand
# $python_gen = "python .\accel\tests\merge_sdf_from_two_hemispheres.py"
# Invoke-Expression -Command $python_gen