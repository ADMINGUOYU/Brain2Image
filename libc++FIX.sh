# Fix of lib C++ problems
# Please run this if you cannot successfully import pandas
ENV_NAME="BRAIN2IMAGE"

# Get conda installation environment path
CONDA_BASE_PATH=$(conda info --base)

# Get env path
ENV_PATH="$CONDA_BASE_PATH/envs/$ENV_NAME"

# cd to env path
cd $ENV_PATH

# Do the fix
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
echo "export OLD_LD_LIBRARY_PATH=\$LD_LIBRARY_PATH" > ./etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=$ENV_PATH/lib:\$LD_LIBRARY_PATH" >> ./etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=\$OLD_LD_LIBRARY_PATH" > ./etc/conda/deactivate.d/env_vars.sh
echo "unset OLD_LD_LIBRARY_PATH" >> ./etc/conda/deactivate.d/env_vars.sh