bo_config_file = 'param_space.yaml'
pheno_path_nan = './ML/pheno_2023bbb_0twins_6traits_mask'
pheno_path = './ML/pheno_20000bbb_6traits'
gen_path = './ML/BBB2023_MD'

from data_cow import load_cow_data
from bo import bo_search

def main():
    ds_tuple = load_cow_data(pheno_path_nan, pheno_path, gen_path, trait=2)
    bo_search(ds_tuple, bo_config_file)

if __name__ == "__main__":
    main()