from nisar_water.round3_scenes import build_scenes
if __name__=='__main__':
    result=build_scenes('data/round3','data/round3-prepared','../nisar-landcover-phase1')
    print('Prepared',len(result['scenes']),'scenes; test metrics remain uninspected')

