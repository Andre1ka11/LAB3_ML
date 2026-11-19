"""Four-model defense figures. Missing human scores are deliberately gray."""
import os
import numpy as np


def figures(root,out,accuracy,classification,hypotheses,paired,costs,code,status,baselines):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=list(dict.fromkeys(r['model'] for r in accuracy));dest=out/'figures';dest.mkdir(exist_ok=True)
    def save(name):plt.tight_layout();plt.savefig(dest/(name+'.png'),dpi=160,bbox_inches='tight');plt.close()
    fig,ax=plt.subplots(figsize=(9,4))
    data=np.array([[next((r['accuracy'] for r in accuracy if r['model']==m and r['type']==t),np.nan) for t in 'ABCDF'] for m in models])
    cmap=plt.get_cmap('YlGn').copy();cmap.set_bad('#e3e5e8');im=ax.imshow(data,vmin=0,vmax=1,cmap=cmap,aspect='auto')
    ax.set(xticks=range(5),xticklabels=['A / 8','B verdict / 8','C expert / 7','D strict / 8','F / 8'],yticks=range(4),yticklabels=models,title='C: full rubric; D: primary contract + execution + metrics + methodology')
    for (i,j),v in np.ndenumerate(data):ax.text(j,i,'pending' if np.isnan(v) else f'{v:.1%}',ha='center',va='center')
    fig.colorbar(im,ax=ax);save('heatmap')
    fig,ax=plt.subplots(figsize=(9,4))
    rows=[next(r for r in accuracy if r['model']==m and r['type']=='ABF') for m in models]
    ax.errorbar(range(4),[r['accuracy'] for r in rows],yerr=[[r['accuracy']-r['ci_low'] for r in rows],[r['ci_high']-r['accuracy'] for r in rows]],fmt='o',capsize=6)
    ax.set(xticks=range(4),xticklabels=models,ylim=(-.05,1.05),ylabel='Accuracy on 24 A/B/F tasks',title='95% percentile bootstrap; 1000 draws, seed 42');save('accuracy_ci')
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,key,title in zip(axes,['f1_minority','balanced_accuracy','coverage'],['Effective F1 class 1','Balanced accuracy','Coverage']):
        for m in models:
            rs=[r for r in classification if r['model']==m];ax.plot([r['shots'] for r in rs],[r[key] for r in rs],'o-',label=m)
        if key!='coverage':
            for name in ['LogisticRegression','GaussianNB']:
                rs=[r for r in baselines if r['model']==name];ax.plot([r['shots'] for r in rs],[r[key] for r in rs],'--',label=name)
        ax.set(title=title,xlabel='Shots',xticks=[0,5,20],ylim=(-.03,1.05))
    axes[0].legend(fontsize=8);save('classification')
    fig,axes=plt.subplots(1,4,figsize=(14,3.5))
    for ax,m in zip(axes,models):
        rs=[r for r in classification if r['model']==m]
        ax.plot([r['shots'] for r in rs],[r['accuracy'] for r in rs],'o-',label='All 100 rows')
        ax.plot([r['shots'] for r in rs],[np.nan if r['selective_accuracy'] is None else r['selective_accuracy'] for r in rs],'s--',label='Answered subset')
        ax.set(title=m,xlabel='Shots',ylim=(-.04,1.04),xticks=[0,5,20])
    axes[0].legend(fontsize=8);save('selective_accuracy')
    fig,axes=plt.subplots(1,4,figsize=(14,3.5))
    for ax,m in zip(axes,models):
        bins=[r for r in hypotheses['H10'][m]['bins'] if r['n']]
        ax.plot([0,1],[0,1],'--',color='gray');ax.scatter([r['mean_confidence'] for r in bins],[r['accuracy'] for r in bins],s=[r['n']*8 for r in bins])
        ax.set(xlim=(0,1.03),ylim=(0,1.03),xlabel='Confidence',title=m)
    axes[0].set_ylabel('Observed accuracy');save('reliability')
    fig,ax=plt.subplots(figsize=(9,4))
    for i,m in enumerate(models):
        h=hypotheses['H10'][m];ax.errorbar([i],[h['gap']],yerr=[[h['gap']-h['ci_low']],[h['ci_high']-h['gap']]],fmt='o',capsize=5)
    ax.axhline(0,color='gray');ax.set(xticks=range(4),xticklabels=models,ylabel='Mean confidence minus accuracy',title='H10: 95% cluster bootstrap intervals');save('h10_gap')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,shots in zip(axes,[5,20]):
        for i,m in enumerate(models):
            h=next(r for r in hypotheses['H12'][m]['comparisons'] if r['shots']==shots);d=h['delta_f1_llm_minus_lr']
            ax.errorbar([i],[d],yerr=[[d-h['one_sided_95_lower']],[h['one_sided_95_upper']-d]],fmt='o',capsize=5)
        ax.axhline(0,color='gray');ax.axhline(-.05,color='red',ls='--');ax.set(xticks=range(4),xticklabels=models,title=f'H12: {shots} shots',ylabel='F1 LLM minus F1 LR');ax.tick_params(axis='x',labelrotation=15)
    save('h12_delta')
    fig,ax=plt.subplots(figsize=(7,3.5));h=hypotheses['H5_primary_Yandex_vs_Qwen'];d=h['statistic_gap_A_minus_gap_B']
    ax.errorbar([0],[d],yerr=[[d-h['ci_low']],[h['ci_high']-d]],fmt='o',capsize=6);ax.axhline(0,color='gray');ax.set(xticks=[0],xticklabels=['(Yandex-Qwen) A minus (Yandex-Qwen) B'],ylabel='Difference in accuracy gaps',title='Primary H5; 95% paired bootstrap interval');save('h5_gap')
    fig,ax=plt.subplots(figsize=(9,4))
    for r in costs:
        acc=next(x['accuracy'] for x in accuracy if x['model']==r['model'] and x['type']=='ABF')
        ax.scatter([r['ABF_cost_rub']],[acc],s=60);ax.annotate(r['model'],(r['ABF_cost_rub'],acc),xytext=(7,-13 if r['model']=='GigaChat-2' else 7),textcoords='offset points')
    ax.set(ylim=(0,1.12),xlabel='Tariff-equivalent API cost for 24 A/B/F tasks (RUB)',ylabel='Accuracy',title='Local zero means API charge only; hardware costs unknown');ax.margins(x=.2);save('cost_accuracy')
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(4)
    for offset,key,label in [(-.25,'pass_at_1','Process/end-to-end pass'),(0,'output_is_json','stdout JSON'),(.25,'metrics_match_reference','Numeric metrics match')]:
        ax.bar(x+offset,[sum(bool(r[key]) for r in code if r['model']==m)/8 for m in models],.25,label=label)
    ax.set(xticks=x,xticklabels=models,ylim=(0,1.18),title='D primary responses; Giga diagnostic extraction excluded');ax.legend(fontsize=8);save('code_execution')
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,m in zip(axes.flat,models):
        matrix=next(r['confusion_0_1_abstain'] for r in classification if r['model']==m and r['shots']==20)
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=65,aspect='auto');ax.set(xticks=[0,1,2],xticklabels=['Pred 0','Pred 1','No answer'],yticks=[0,1],yticklabels=['True 0','True 1'],title=m+' 20-shot')
        for (i,j),v in np.ndenumerate(np.array(matrix)):ax.text(j,i,str(v),ha='center',va='center')
    save('confusion_20')
