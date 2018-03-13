import messlkeras as mk

def prep_chime3_lists():
    # given an experiment type (iaf, psf, msa, psa), return the input and target lists for keras
    # as : return ALL tr_lists, dt_lists, et_lists (for chime3 data)
    # with:
    # tr_lists = [chime_spects_noisy_tr, messl_soft_masks_tr, iaf_masks_tr, psf_masks_tr, messl_spects_mvdr_tr]
    # dt_lists = [chime_spects_noisy_dt, messl_soft_masks_dt, iaf_masks_dt, psf_masks_dt, messl_spects_mvdr_dt]
    # et_lists = [chime_spects_noisy_et, messl_soft_masks_et, iaf_masks_et, psf_masks_et, messl_spects_mvdr_et]
        
    # hardcoded # directories of files to work on
    chime_dir = "/home/data/CHiME3/data/audio/16kHz/local/"
    messl_soft_masks_dir = "/scratch/proj/messl/vanilla_MESSL_output/MESSL_softmasks/" # soft masks 
    
    # training
    tr_reg_exp = ".*tr05_.*_((simu)|(real)).*"
    chime_spects_noisy_tr = mk.prep_list_for_keras(chime_dir, "messl-spects-noisy"+tr_reg_exp, verbose=False)
    messl_soft_masks_tr = mk.prep_list_for_keras(messl_soft_masks_dir, tr_reg_exp, verbose=False)

    iaf_masks_tr = mk.prep_list_for_keras(chime_dir, "mask.*ideal_amplitude"+tr_reg_exp, verbose=False)
    psf_masks_tr = mk.prep_list_for_keras(chime_dir, "mask.*phase_sensitive"+tr_reg_exp, verbose=False)
    messl_spects_mvdr_tr = mk.prep_list_for_keras(chime_dir, "messl-spects-mvdr-cleaned"+tr_reg_exp, verbose=False)

    # validation
    dt_reg_exp = ".*dt05_.*_((simu)|(real)).*"
    chime_spects_noisy_dt = mk.prep_list_for_keras(chime_dir, "messl-spects-noisy"+dt_reg_exp, verbose=False)
    messl_soft_masks_dt = mk.prep_list_for_keras(messl_soft_masks_dir, dt_reg_exp, verbose=False)

    iaf_masks_dt = mk.prep_list_for_keras(chime_dir, "mask.*ideal_amplitude"+dt_reg_exp, verbose=False)
    psf_masks_dt = mk.prep_list_for_keras(chime_dir, "mask.*phase_sensitive"+dt_reg_exp, verbose=False)
    messl_spects_mvdr_dt = mk.prep_list_for_keras(chime_dir, "messl-spects-mvdr-cleaned"+dt_reg_exp, verbose=False)

    # testing
    et_reg_exp = ".*et05_.*_((simu)|(real)).*"
    chime_spects_noisy_et = mk.prep_list_for_keras(chime_dir, "messl-spects-noisy"+et_reg_exp, verbose=False)
    messl_soft_masks_et = mk.prep_list_for_keras(messl_soft_masks_dir, et_reg_exp, verbose=False)

    iaf_masks_et = mk.prep_list_for_keras(chime_dir, "mask.*ideal_amplitude"+et_reg_exp, verbose=False)
    psf_masks_et = mk.prep_list_for_keras(chime_dir, "mask.*phase_sensitive"+et_reg_exp, verbose=False)
    messl_spects_mvdr_et = mk.prep_list_for_keras(chime_dir, "messl-spects-mvdr-cleaned"+et_reg_exp, verbose=False)


    #check that the filenames all match
    if ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in messl_soft_masks_tr]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in iaf_masks_tr]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in psf_masks_tr]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_tr] == [x.split('/')[-1] for x in messl_spects_mvdr_tr]) :
        print("Number of training files:", len(chime_spects_noisy_tr))
    else:
        raise Exception("Training filenames for inputs and targets do not match! Exiting")
        
    #check that the filenames all match
    if ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in messl_soft_masks_dt]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in iaf_masks_dt]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in psf_masks_dt]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_dt] == [x.split('/')[-1] for x in messl_spects_mvdr_dt]) :
        print("Number of validation files:", len(chime_spects_noisy_dt))
    else:
        raise Exception("Validation filenames for inputs and targets do not match! Exiting")
        
    #check that the filenames all match
    if ([x.split('/')[-1] for x in chime_spects_noisy_et] == [x.split('/')[-1] for x in messl_soft_masks_et]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_et] == [x.split('/')[-1] for x in iaf_masks_et]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_et] == [x.split('/')[-1] for x in psf_masks_et]) \
        and ([x.split('/')[-1] for x in chime_spects_noisy_et] == [x.split('/')[-1] for x in messl_spects_mvdr_et]) :
        print("Number of testing files:", len(chime_spects_noisy_et))
    else:
        raise Exception("Testing filenames for inputs and targets do not match! Exiting")

    
    # to return:
    tr_lists = [chime_spects_noisy_tr, messl_soft_masks_tr, iaf_masks_tr, psf_masks_tr, messl_spects_mvdr_tr]
    dt_lists = [chime_spects_noisy_dt, messl_soft_masks_dt, iaf_masks_dt, psf_masks_dt, messl_spects_mvdr_dt]
    et_lists = [chime_spects_noisy_et, messl_soft_masks_et, iaf_masks_et, psf_masks_et, messl_spects_mvdr_et]
        
    return tr_lists, dt_lists, et_lists




    # # validation
    # dt_reg_exp = ".*dt05_.*_((simu)|(real)).*"
    # chime_spects_noisy_dt = mk.prep_list_for_keras(chime_dir, "messl-spects-noisy"+dt_reg_exp, verbose=False)
    # messl_soft_masks__dt = mk.prep_list_for_keras(messl_soft_masks_dir, dt_reg_exp, verbose=False)

    # if exp_type == 'iaf':
    #     target_list_dt = mk.prep_list_for_keras(chime_dir, "mask.*ideal_amplitude"+dt_reg_exp, verbose=False)
    # elif exp_type == 'psf':
    #     target_list_dt = mk.prep_list_for_keras(chime_dir, "mask.*phase_sensitive"+dt_reg_exp, verbose=False)
    # elif exp_type in ['msa', 'psa']:
    #     target_list_dt = mk.prep_list_for_keras(chime_dir, "messl-spects-mvdr-cleaned"+dt_reg_exp, verbose=False)

    # # testing
    # et_reg_exp = ".*et05_.*_((simu)|(real)).*"
    # chime_spects_noisy_et = mk.prep_list_for_keras(chime_dir, "messl-spects-noisy"+et_reg_exp, verbose=False)
    # messl_soft_masks__et = mk.prep_list_for_keras(messl_soft_masks_dir, et_reg_exp, verbose=False)

    # if exp_type == 'iaf':
    #     target_list_et = mk.prep_list_for_keras(chime_dir, "mask.*ideal_amplitude"+et_reg_exp, verbose=False)
    # elif exp_type == 'psf':
    #     target_list_et = mk.prep_list_for_keras(chime_dir, "mask.*phase_sensitive"+et_reg_exp, verbose=False)
    # elif exp_type in ['msa', 'psa']:
    #     target_list_et = mk.prep_list_for_keras(chime_dir, "messl-spects-mvdr-cleaned"+et_reg_exp, verbose=False)