"""Task 3: redundancy analysis and parsimonious logistic selection.
Run: python task3_multivariate_selection.py --input AppData.xlsx --task2-dir task2_output --output-dir task3_output
All preprocessing and WoE maps are fitted inside each CV training fold. Validation from Task 1 remains untouched.
"""
from __future__ import annotations
import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import task2_univariate_analysis as t2
from task1_data_preparation import add_initial_features, stratified_split, validate_input, SEED

CS = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
FOLDS = 5
ALPHA = 0.5

def raw_data(path):
    d = pd.read_excel(path); validate_input(d); tr, va = stratified_split(d)
    return add_initial_features(tr), add_initial_features(va)

def fit_encoder(frame, y, variables):
    maps, kinds = {}, {}
    min_n = max(1, int(math.ceil(t2.MIN_BIN_SHARE * len(frame))))
    for v in variables:
        kind = t2.feature_kind(frame[v]); kinds[v] = kind
        ass, meta = t2.make_fine_bins(frame, v, kind)
        rec = t2.count_bins(ass, y, meta)
        groups = [{"fine_ids":[r["bin_id"]], "labels":[r["bin_label"]], "n":r["n"], "good":r["good"], "bad":r["bad"], "numeric_low":r["numeric_low"], "numeric_high":r["numeric_high"], "is_missing":r["is_missing"], "is_special":r["is_special"], "fine_order":r["fine_order"]} for r in rec]
        log=[]; groups=t2._merge_until_support(groups,kind,min_n,t2.MAX_COARSE_BINS,v,log)
        direction=t2.EXPECTED_MONOTONICITY.get(v)
        if direction: groups=t2._merge_monotonic_violations(groups,kind,direction,v,log)
        fine_to_group={fid:i for i,g in enumerate(groups) for fid in g["fine_ids"]}
        fine_meta={m["bin_id"]:m for m in meta}
        grp_counts=[]
        for i,g in enumerate(groups):
            rows=[r for r in rec if r["bin_id"] in g["fine_ids"]]
            good=sum(r["good"] for r in rows); bad=sum(r["bad"] for r in rows)
            grp_counts.append((good,bad))
        total_g=sum(a for a,b in grp_counts); total_b=sum(b for a,b in grp_counts); k=len(groups)
        smooth=any(a==0 or b==0 for a,b in grp_counts)
        alpha=ALPHA if smooth else 0.0
        woe=[math.log(((a+alpha)/(total_g+alpha*k))/((b+alpha)/(total_b+alpha*k))) for a,b in grp_counts]
        maps[v]=(kind,meta,fine_meta,fine_to_group,woe)
    return maps

def transform(frame, maps):
    out=pd.DataFrame(index=frame.index)
    for v,(kind,meta,fine_meta,fine_to_group,woe) in maps.items():
        x=frame[v]; ids=pd.Series(index=frame.index,dtype="object")
        if kind=="continuous":
            for m in meta:
                bid=m["bin_id"]
                if m["is_special"]:
                    mask=frame["EMPLOYMENT_SPECIAL_CODE"].eq(1).fillna(False) if v=="EMPLOYMENT_YEARS" else pd.Series(False,index=frame.index)
                elif m["is_missing"]: mask=x.isna() & ((~frame["EMPLOYMENT_SPECIAL_CODE"].eq(1).fillna(False)) if v=="EMPLOYMENT_YEARS" else True)
                else:
                    lo,hi=m["numeric_low"],m["numeric_high"]
                    mask=x.notna() & (x<=hi) & ((x>=lo) if m["fine_order"]==0 else (x>lo))
                    if v=="EMPLOYMENT_YEARS": mask &= ~frame["EMPLOYMENT_SPECIAL_CODE"].eq(1).fillna(False)
                ids.loc[mask]=bid
        elif kind=="ordered_discrete":
            for m in meta:
                if m["is_missing"]: mask=x.isna()
                else: mask=x.eq(float(m["category_values"])).fillna(False)
                ids.loc[mask]=m["bin_id"]
        else:
            tok=x.map(lambda z:t2.MISSING_TOKEN if pd.isna(z) else str(z))
            for m in meta: ids.loc[tok.eq(m["category_values"])]=m["bin_id"]
        vals=ids.map(fine_to_group).map(lambda g: woe[int(g)] if pd.notna(g) else 0.0).astype(float)
        out[v]=vals
    return out

def cramers_v(a,b):
    tab=pd.crosstab(a.fillna("<MISSING>").astype(str),b.fillna("<MISSING>").astype(str))
    if min(tab.shape)<2:return np.nan
    chi2,p,_,_=chi2_contingency(tab, correction=False); n=tab.to_numpy().sum(); r,k=tab.shape
    phi2=chi2/n; corr=max(0,phi2-(k-1)*(r-1)/(n-1)); rr=r-(r-1)**2/(n-1); kk=k-(k-1)**2/(n-1)
    return math.sqrt(corr/min(kk-1,rr-1)) if min(kk,rr)>1 else np.nan

def scores(y,p): return {"auc":roc_auc_score(y,p),"gini":2*roc_auc_score(y,p)-1,"log_loss":log_loss(y,p,labels=[0,1])}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',default='AppData.xlsx'); ap.add_argument('--task2-dir',default='task2_output'); ap.add_argument('--output-dir',default='task3_output'); args=ap.parse_args()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    train, _validation=raw_data(args.input); y=train.TARGET.astype(int)
    shortlist=pd.read_csv(Path(args.task2_dir)/'shortlist.csv')['variable'].tolist()
    shortlist=[v for v in shortlist if v in train.columns and v not in {'SK_ID_CURR','TARGET','EMPLOYMENT_SPECIAL_CODE'}]
    # Full-training diagnostics only; used for redundancy review, not CV scoring.
    enc=fit_encoder(train,y,shortlist); X=transform(train,enc)
    corr=[]
    for i,a in enumerate(shortlist):
      for b in shortlist[i+1:]:
        pr=pearsonr(X[a],X[b]).statistic; sr=spearmanr(X[a],X[b]).statistic
        corr.append({'variable_1':a,'variable_2':b,'pearson_woe':pr,'spearman_woe':sr,'abs_spearman':abs(sr),'review_flag':abs(sr)>=.70})
    corr=pd.DataFrame(corr).sort_values('abs_spearman',ascending=False); corr.to_csv(out/'woe_correlations.csv',index=False)
    nominal=[v for v in shortlist if t2.feature_kind(train[v])=='categorical']
    cv=[]
    for i,a in enumerate(nominal):
      for b in nominal[i+1:]: cv.append({'variable_1':a,'variable_2':b,'cramers_v':cramers_v(train[a],train[b])})
    pd.DataFrame(cv).sort_values('cramers_v',ascending=False).to_csv(out/'cramers_v.csv',index=False)
    # VIF on training WoE matrix, interpreted only as a diagnostic.
    z=X.replace([np.inf,-np.inf],np.nan).fillna(0); z=(z-z.mean())/z.std(ddof=0).replace(0,1)
    vif=[]
    for j,v in enumerate(z.columns):
      try: val=variance_inflation_factor(z.to_numpy(),j)
      except Exception: val=np.inf
      vif.append({'variable':v,'VIF':val})
    pd.DataFrame(vif).sort_values('VIF',ascending=False).to_csv(out/'vif.csv',index=False)
    # Stratified CV: refit bins/WoE inside each fold, standardize on fold training only.
    scenarios={'all_candidates':list(range(len(shortlist))),'exclude_gender':[j for j,v in enumerate(shortlist) if v!='CODE_GENDER'],'drop_APARTMENTS_AVG':[j for j,v in enumerate(shortlist) if v!='APARTMENTS_AVG'],'exclude_gender_and_apartments':[j for j,v in enumerate(shortlist) if v not in {'CODE_GENDER','APARTMENTS_AVG'}]}
    skf=StratifiedKFold(n_splits=FOLDS,shuffle=True,random_state=SEED); fold_results=[]; full_results=[]
    for fold,(it,iv) in enumerate(skf.split(train,y),1):
      a=train.iloc[it]; b=train.iloc[iv]; ya=y.iloc[it]; yb=y.iloc[iv]
      em=fit_encoder(a,ya,shortlist); xa=transform(a,em); xb=transform(b,em)
      scaler=StandardScaler(); xa_s=scaler.fit_transform(xa); xb_s=scaler.transform(xb)
      full=LogisticRegression(C=1e6,solver='lbfgs',max_iter=3000).fit(xa_s,ya)
      full_results.append({'fold':fold,'n_variables':len(shortlist),**scores(yb,full.predict_proba(xb_s)[:,1])})
      for C in CS:
       for scenario,cols in scenarios.items():
        mod=LogisticRegression(C=C,l1_ratio=1.0,solver='liblinear',max_iter=3000).fit(xa_s[:,cols],ya)
        pred=mod.predict_proba(xb_s[:,cols])[:,1]
        fold_results.append({'scenario':scenario,'fold':fold,'C':C,'n_selected':int((np.abs(mod.coef_[0])>1e-8).sum()),**scores(yb,pred)})
    folds=pd.DataFrame(fold_results); folds.to_csv(out/'lasso_cv_fold_results.csv',index=False)
    full_df=pd.DataFrame(full_results); full_df.to_csv(out/'full_shortlist_cv.csv',index=False)
    summary=folds.groupby(['scenario','C']).agg(mean_gini=('gini','mean'),sd_gini=('gini','std'),mean_auc=('auc','mean'),mean_log_loss=('log_loss','mean'),mean_variables=('n_selected','mean')).reset_index()
    summary.to_csv(out/'lasso_cv_summary.csv',index=False)
    # 1-SE choice: largest penalty (smallest C) whose mean Gini is within one SE of maximum.
    selected_info=[]; comparison=[]
    for scenario in summary.scenario.unique():
      ss=summary[summary.scenario==scenario].copy(); best=ss.loc[ss.mean_gini.idxmax()]; threshold=best.mean_gini-best.sd_gini/math.sqrt(FOLDS)
      eligible=ss[ss.mean_gini>=threshold].sort_values(['mean_variables','C'],ascending=[True,True]); chosen=eligible.iloc[0] if len(eligible) else best
      chosenC=float(chosen.C); cols=scenarios[scenario]
      final_enc=fit_encoder(train,y,shortlist); xf=transform(train,final_enc); sc=StandardScaler(); xs=sc.fit_transform(xf.iloc[:,cols])
      model=LogisticRegression(C=chosenC,l1_ratio=1.0,solver='liblinear',max_iter=3000).fit(xs,y)
      kept=[shortlist[j] for j,k in zip(cols,model.coef_[0]) if abs(k)>1e-8]
      selected_info += [{'scenario':scenario,'C':chosenC,'variable':shortlist[j],'standardized_lasso_beta':float(k)} for j,k in zip(cols,model.coef_[0]) if abs(k)>1e-8]
      # Unpenalized refit for interpretable coefficients and secondary AIC/BIC.
      aic=bic=np.nan; wald=np.nan
      if kept:
       xr=xf[kept].copy(); xr=(xr-xr.mean())/xr.std(ddof=0).replace(0,1); xr=sm.add_constant(xr,has_constant='add')
       try:
        fit=sm.Logit(y,xr).fit(disp=False,maxiter=500)
        aic=float(fit.aic); bic=float(fit.bic); wald=float(fit.llr_pvalue)
       except Exception: pass
      rec=summary[(summary.scenario==scenario)&(summary.C==chosenC)].iloc[0]
      comparison.append({'scenario':scenario,'selected_C':chosenC,'selection_rule':'1-SE relative to best mean CV Gini','n_variables_selected':len(kept),'cv_mean_gini':float(rec.mean_gini),'cv_sd_gini':float(rec.sd_gini),'cv_mean_auc':float(rec.mean_auc),'cv_mean_log_loss':float(rec.mean_log_loss),'unpenalized_refit_AIC_secondary':aic,'unpenalized_refit_BIC_secondary':bic,'refit_LR_p_value':wald,'variables':', '.join(kept)})
    pd.DataFrame(selected_info,columns=['scenario','C','variable','standardized_lasso_beta']).to_csv(out/'selected_variables.csv',index=False)
    pd.DataFrame(comparison).to_csv(out/'model_comparison.csv',index=False)
    corr.head(20).to_csv(out/'high_correlation_review.csv',index=False)
    notes=[f'Task 3; seed={SEED}; training n={len(train)}; sample bad rate={y.mean():.4%}; candidate predictors={len(shortlist)}; stratified CV folds={FOLDS}.','All fold-specific classing/WoE maps and scaling are refitted within fold training data. Task 1 validation is not used.','Correlation flags (|Spearman WoE| >= 0.70) and VIF are review indicators, not automatic deletion rules. Cramers V is reported among categorical shortlisted predictors.','L1 logistic uses standardized WoE and target TARGET=1 (bad). Negative coefficients are generally consistent with good-over-bad WoE. CODE_GENDER is run both in and out of the shortlist as a fairness sensitivity; inclusion requires explicit fairness/legal review.','The one-standard-error rule seeks a smaller model with CV Gini within one standard error of the best mean. AIC/BIC are secondary in-sample refit summaries. CV metrics reflect the stratified 33.3% sample bad rate; they are not population-calibrated PD metrics.']
    (out/'task3_notes.txt').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print('Task 3 complete. Candidate count:',len(shortlist),'training n:',len(train))
    print(pd.DataFrame(comparison).to_string(index=False))
    print('Results:',out.resolve())
if __name__=='__main__': main()





