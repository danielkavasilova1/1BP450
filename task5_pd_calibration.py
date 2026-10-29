"""Task 5: recalibrate PDs to the 8% population prior, create grades A-J, and apply a proposed policy.
Run: python task5_pd_calibration.py --input AppData.xlsx --task3-dir task3_output --output-dir task5_output
Training-only: bins/WoE/model and population-weighted score cutoffs. Validation is scored and summarized only.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import task3_multivariate_selection as t3
from task1_data_preparation import SEED

POPULATION_BAD_RATE=0.08
BASE_SCORE=600.0
BASE_GOOD_BAD_ODDS=50.0
PDO=20.0
N_GRADES=10
AUTO_APPROVE_PD=0.079
HARD_DECLINE_PD=0.14
SCENARIO="exclude_gender_and_apartments"

def sigmoid(x): return 1/(1+np.exp(-np.clip(np.asarray(x,dtype=float),-35,35)))
def weighted_mean(x,w): return float(np.average(np.asarray(x,dtype=float),weights=np.asarray(w,dtype=float)))
def weights_for(y, rate):
    y=np.asarray(y,dtype=int); rho=float(y.mean())
    return np.where(y==1,rate/rho,(1-rate)/(1-rho))
def weighted_quantile(x,w,q):
    x=np.asarray(x,dtype=float); w=np.asarray(w,dtype=float); ix=np.argsort(x,kind='mergesort'); x,w=x[ix],w[ix]
    cdf=(np.cumsum(w)-.5*w)/w.sum()
    return np.interp(q,cdf,x,left=x[0],right=x[-1])
def decision(pd_value):
    if pd_value >= HARD_DECLINE_PD: return 'Hard decline'
    if pd_value >= AUTO_APPROVE_PD: return 'Manual review'
    return 'Auto approve'
def letter_grade(rank): return chr(64+int(rank))
def score_at_pd(p, factor):
    logit=math.log(p/(1-p))
    return BASE_SCORE-factor*(logit+math.log(BASE_GOOD_BAD_ODDS))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='AppData.xlsx'); ap.add_argument('--task3-dir',default='task3_output'); ap.add_argument('--output-dir',default='task5_output'); ap.add_argument('--population-bad-rate',type=float,default=POPULATION_BAD_RATE); ap.add_argument('--n-grades',type=int,default=N_GRADES,choices=range(7,11)); a=ap.parse_args()
    if not 0<a.population_bad_rate<1: raise ValueError('Population bad rate must be between 0 and 1.')
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    train,val=t3.raw_data(a.input); y=train.TARGET.astype(int); yv=val.TARGET.astype(int)
    shortlist=pd.read_csv(Path(a.task3_dir)/'selected_variables.csv')
    variables=shortlist.loc[shortlist.scenario.eq(SCENARIO),'variable'].drop_duplicates().tolist()
    if not variables: raise ValueError('Task 3 selected variables were not found.')
    enc=t3.fit_encoder(train,y,variables); xt=t3.transform(train,enc); xv=t3.transform(val,enc)
    predictors=[v for v in variables if xt[v].nunique(dropna=False)>1]
    xt=xt[predictors].astype(float); xv=xv[predictors].astype(float)
    fit=sm.Logit(y,sm.add_constant(xt,has_constant='add')).fit(disp=False,maxiter=1000)
    rho=float(y.mean()); pi=a.population_bad_rate
    shift=math.log(pi/(1-pi))-math.log(rho/(1-rho))
    eta_t=float(fit.params['const'])+xt.dot(fit.params[predictors]); eta_v=float(fit.params['const'])+xv.dot(fit.params[predictors])
    pd_t_sample=sigmoid(eta_t); pd_v_sample=sigmoid(eta_v); pd_t=sigmoid(eta_t+shift); pd_v=sigmoid(eta_v+shift)
    factor=PDO/math.log(2); base=BASE_SCORE-factor*(float(fit.params['const'])+shift+math.log(BASE_GOOD_BAD_ODDS))
    def score(x):
        z=pd.Series(base,index=x.index,dtype=float)
        for v in predictors: z += -factor*float(fit.params[v])*x[v]
        return z
    score_t=score(xt); score_v=score(xv)
    wt=weights_for(y,pi); wv=weights_for(yv,pi)
    cutpoints=np.unique(weighted_quantile(score_t,wt,np.linspace(0,1,a.n_grades+1)))
    ngrades=len(cutpoints)-1
    if ngrades<7: raise ValueError(f'Only {ngrades} distinct score bands; need at least 7.')
    # score_band=1 is worst/lowest score; reverse rank so Grade A is best/highest score.
    def score_band(score_values): return np.clip(np.searchsorted(cutpoints[1:-1],np.asarray(score_values,dtype=float),side='left')+1,1,ngrades).astype(int)
    band_t=score_band(score_t); band_v=score_band(score_v)
    rank_t=ngrades-band_t+1; rank_v=ngrades-band_v+1
    grade_t=np.array([letter_grade(r) for r in rank_t]); grade_v=np.array([letter_grade(r) for r in rank_v])
    auto_score=score_at_pd(AUTO_APPROVE_PD,factor); decline_score=score_at_pd(HARD_DECLINE_PD,factor)
    cutrows=[]
    for rank in range(1,ngrades+1):
        band=ngrades-rank+1; low,high=cutpoints[band-1],cutpoints[band]
        share=weighted_mean((rank_t==rank).astype(float),wt)
        mask=rank_t==rank; meanpd=weighted_mean(pd_t[mask],wt[mask]) if mask.any() else float('nan')
        cutrows.append({'grade':letter_grade(rank),'grade_rank_1_best':rank,'score_lower_inclusive':low,'score_upper_inclusive':high,'weighted_training_share':share,'training_mean_calibrated_PD':meanpd})
    pd.DataFrame(cutrows).to_csv(out/'grade_cutoffs.csv',index=False)
    decisions=np.array([decision(v) for v in pd_v])
    scored=val[['SK_ID_CURR','TARGET']].copy(); scored['score_unrounded']=score_v; scored['score']=np.rint(score_v).astype(int); scored['pd_sample']=pd_v_sample; scored['calibrated_pd']=pd_v; scored['grade']=grade_v; scored['grade_rank_1_best']=rank_v; scored['proposed_decision']=decisions; scored['population_weight']=wv
    scored.to_csv(out/'validation_scored_with_grades.csv',index=False)
    graderows=[]
    for rank in range(1,ngrades+1):
        m=rank_v==rank; g=letter_grade(rank); band=ngrades-rank+1
        if not m.any(): graderows.append({'grade':g,'grade_rank_1_best':rank,'validation_clients':0}); continue
        meanpd=weighted_mean(pd_v[m],wv[m]); minpd=float(pd_v[m].min()); maxpd=float(pd_v[m].max())
        if maxpd < AUTO_APPROVE_PD: action='Auto approve'
        elif minpd >= HARD_DECLINE_PD: action='Hard decline'
        elif minpd >= AUTO_APPROVE_PD and maxpd < HARD_DECLINE_PD: action='Manual review'
        elif minpd < AUTO_APPROVE_PD: action='Mixed: apply PD cutoffs (approve/review)'
        else: action='Mixed: apply PD cutoffs (review/decline)'
        graderows.append({'grade':g,'grade_rank_1_best':rank,'risk_order':'best' if rank==1 else ('worst' if rank==ngrades else 'intermediate'),'score_lower_inclusive':float(cutpoints[band-1]),'score_upper_inclusive':float(cutpoints[band]),'validation_clients':int(m.sum()),'population_equivalent_clients':float(wv[m].sum()),'average_calibrated_PD':meanpd,'minimum_calibrated_PD':minpd,'maximum_calibrated_PD':maxpd,'average_population_weighted_default_rate':weighted_mean(yv.to_numpy()[m],wv[m]),'validation_sample_default_rate':float(yv.to_numpy()[m].mean()),'suggested_decision_by_grade_average':action})
    grade_table=pd.DataFrame(graderows); grade_table.to_csv(out/'validation_grade_calibration.csv',index=False)
    decrows=[]
    for action in ['Auto approve','Manual review','Hard decline']:
        m=decisions==action
        if not m.any(): decrows.append({'decision':action,'validation_clients':0}); continue
        decrows.append({'decision':action,'validation_clients':int(m.sum()),'validation_share':float(m.mean()),'population_equivalent_clients':float(wv[m].sum()),'average_calibrated_PD':weighted_mean(pd_v[m],wv[m]),'population_weighted_observed_default_rate':weighted_mean(yv.to_numpy()[m],wv[m])})
    dec_table=pd.DataFrame(decrows); dec_table.to_csv(out/'validation_decision_summary.csv',index=False)
    summary=pd.DataFrame([('Training sample bad rate',rho),('Validation sample bad rate',float(yv.mean())),('Assumed population bad rate',pi),('Logit intercept correction',shift),('Validation population-weighted observed bad rate',weighted_mean(yv,wv)),('Validation population-weighted mean predicted PD',weighted_mean(pd_v,wv)),('Training population-weighted mean predicted PD',weighted_mean(pd_t,wt)),('Number of grades',ngrades),('Score threshold for auto-approve PD',auto_score),('Score threshold for hard-decline PD',decline_score)],columns=['measure','value'])
    summary.to_csv(out/'calibration_summary.csv',index=False)
    notes=[f'Seed={SEED}; training n={len(train)}; validation n={len(val)}.',f'Scenario={SCENARIO}; predictors={", ".join(predictors)}.',f'Training sample bad rate={rho:.8%}; assumed population bad rate={pi:.8%}; logit intercept correction={shift:.10f}.',f'Grade A is best/high score and grade {letter_grade(ngrades)} is worst/low score. {ngrades} population-weighted cutoffs were estimated on training data and frozen for validation.',f'Proposed policy: calibrated PD < {AUTO_APPROVE_PD:.1%} auto-approve (score > {auto_score:.2f}); {AUTO_APPROVE_PD:.1%} <= PD < {HARD_DECLINE_PD:.1%} manual review ({decline_score:.2f} < score <= {auto_score:.2f}); PD >= {HARD_DECLINE_PD:.1%} hard-decline (score <= {decline_score:.2f}).', 'The thresholds are a proposed policy scenario, not an economically optimized cutoff. Optimization requires loan margin, LGD/recovery, exposure, and review/operating costs.', 'Population-equivalent counts are weighted representations, not observed applicant counts. Validation weights reweight the case-enriched sample to the assumed population default rate.', 'The prior correction assumes class-conditional predictor distributions are stable. No validation outcomes were used for model fitting or grade cutoffs.']
    (out/'task5_notes.txt').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print('Task 5 complete | train',len(train),'| validation',len(val),'| grades',ngrades)
    print('Intercept correction',round(shift,8),'| weighted observed rate',round(weighted_mean(yv,wv),6),'| weighted mean PD',round(weighted_mean(pd_v,wv),6))
    print('PD cutoffs: auto <',AUTO_APPROVE_PD,'; hard decline >=',HARD_DECLINE_PD,'| score cutoffs: auto >',round(auto_score,2),'; hard decline <=',round(decline_score,2))
    print(grade_table.to_string(index=False)); print(dec_table.to_string(index=False)); print('Outputs:',out.resolve())
if __name__=='__main__':main()


