"""Post-freeze report only: gold never changes candidates, thresholds, or selected pointer."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from common import ROOT,BASE,ASPECTS,GOALS,NLI_ID,sha,write_json,metrics
from inference import scores,decide,predict,blend


def bootstrap(gold,pred,old,n=2000):
    codes,groups=pd.factorize(gold.review_id)
    weights=np.random.RandomState(20261018).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=n)
    def values(p):
        def matrix(task,classes,mask=None,predclasses=None):
            pc=predclasses or classes;ny=len(classes);npred=len(pc)
            if mask is None:mask=np.ones(len(gold),dtype=bool)
            actual=gold.loc[mask,task].map({v:i for i,v in enumerate(classes)}).to_numpy(int)
            actualp=p.loc[mask,task].map({v:i for i,v in enumerate(pc)}).to_numpy(int)
            grouped=np.zeros((len(groups),ny*npred),dtype=int)
            np.add.at(grouped,(codes[mask],actual*npred+actualp),1)
            return (weights@grouped).reshape(n,ny,npred)
        def mf(cm):
            nc=cm.shape[1];tp=np.diagonal(cm,axis1=1,axis2=2)
            return 2*tp/np.maximum(1,cm.sum(2)+cm.sum(1)[:,:nc])
        sub=matrix('subjectivity',['objective','subjective']);a=matrix('aspect',ASPECTS)
        lab=matrix('label',['objective','negative','positive'])
        pol=matrix('polarity',['negative','positive'],gold.subjectivity.eq('subjective').to_numpy(),['negative','positive','NA'])
        count=np.bincount(codes,minlength=len(groups));correct=np.bincount(codes,weights=(gold.aspect.eq(p.aspect)&gold.label.eq(p.label)).to_numpy(int),minlength=len(groups))
        return {'subjectivity_f1':mf(sub).mean(1),'aspect_f1':mf(a).mean(1),'label_f1':mf(lab).mean(1),
                'polarity_f1':mf(pol).mean(1),'objective_precision':sub[:,0,0]/np.maximum(1,sub[:,:,0].sum(1)),
                'objective_recall':sub[:,0,0]/np.maximum(1,sub[:,0,:].sum(1)),
                'minimum_aspect_f1':mf(a).min(1),'joint_accuracy':(weights@correct)/(weights@count)}
    a,b=values(pred),values(old)
    return {k:{'conditional_95':np.quantile(a[k],[.025,.975]).tolist(),
               'paired_delta_vs_previous_95':np.quantile(a[k]-b[k],[.025,.975]).tolist()} for k in GOALS}


def main(out):
    freeze_path=out/'frozen_candidates.json';frozen=json.loads(freeze_path.read_text());freeze_hash=sha(freeze_path)
    if (out/'gold_audit.json').exists():raise RuntimeError('Preserve the completed audit; no repeated gold optimization')
    for name,digest in frozen['candidates'].items():assert sha(out/name/'model.joblib')==digest
    for file,digest in frozen['scripts'].items():assert sha(file)==digest,'Fitting or inference source changed after freezing'
    for file,digest in frozen.get('input_hashes',{}).items():assert sha(file)==digest,'Weak fitting inputs changed after freezing'
    status={'status':'running','phase':'post-freeze gold audit','started_at':datetime.now(timezone.utc).isoformat(),
            'freeze_sha256':freeze_hash,'selected_without_gold':frozen['selected_without_gold'],'completed':[]}
    def phase(value):
        status.update(phase=value,updated_at=datetime.now(timezone.utc).isoformat());write_json(out/'audit_monitor.json',status)
        print(value,flush=True)
    gold_path=BASE/'inputs/gold_1073.csv'
    phase('Reading gold only after every candidate and selected pointer has been frozen')
    gold=pd.read_csv(gold_path,keep_default_na=False);gold_hash=sha(gold_path)
    # Recompute the retained previous candidate for a direct paired comparison.
    prevdir=ROOT/'outputs/c_tuning_20261018';sys.path.insert(0,str(prevdir))
    from tuned_predict import predict as previous_predict
    previnfo=json.loads((prevdir/'best.json').read_text());previous=previous_predict(joblib.load(previnfo['model']),gold.text.tolist())
    oldm=metrics(gold,previous)
    cache={};predictions={};results=[];latency={}
    for name,digest in frozen['candidates'].items():
        phase(name+': scoring frozen candidate')
        model=joblib.load(out/name/'model.joblib');start=time.perf_counter()
        if model['kind']=='fusion':
            details=json.loads((out/name/'metrics.json').read_text())['details']
            sc=blend(cache[details['weak_source']],cache[details['nli_source']],details['nli_weight'])
        else:
            def progress(done,total):
                if done%128==0 or done==total:phase(f'{name}: local inference {done}/{total} unique texts')
            sc=scores(model,gold.text.tolist(),progress)
        elapsed=time.perf_counter()-start
        pred=decide(sc,gold.text.tolist(),model['decision_config'])
        m=metrics(gold,pred);fail={k:{'actual':m[k],'target':v} for k,v in GOALS.items() if m[k]<v}
        r={'name':name,'metrics':m,'failed_goals':fail,'all_point_targets_met':not fail,
           'selected_before_gold':name==frozen['selected_without_gold'],'gold_changed_selection':False,
           'score_seconds':elapsed,'fused_scores_reused':model['kind']=='fusion'}
        write_json(out/name/'gold_metrics.json',r)
        np.savez_compressed(out/name/'gold_scores.npz',**sc)
        gold.add_prefix('gold_').join(pred.add_prefix('pred_')).to_json(out/name/'gold_predictions.jsonl',orient='records',lines=True,force_ascii=False)
        cache[name]=sc;predictions[name]=pred;results.append(r);latency[name]=elapsed
        status['completed'].append(name);phase(name+': '+json.dumps({k:round(m[k],4) for k in GOALS}))
    selected=frozen['selected_without_gold'];chosen=next(r for r in results if r['name']==selected)
    phase('Verifying serialized selected bundle on every gold row and computing paired uncertainty')
    restored=joblib.load(out/selected/'model.joblib')
    restored_pred=predict(restored,gold.text.tolist())
    pd.testing.assert_frame_equal(restored_pred,predictions[selected])
    m=chosen['metrics'];checks={}
    for task,classes in [('aspect',ASPECTS),('subjectivity',['objective','subjective']),('label',['objective','negative','positive'])]:
        report=classification_report(gold[task],restored_pred[task],labels=classes,output_dict=True,zero_division=0)
        assert abs(report['macro avg']['f1-score']-m[task+'_f1'])<1e-12
        checks[task]={'report':report,'classes':classes,'confusion_matrix':confusion_matrix(gold[task],restored_pred[task],labels=classes).tolist()}
    intervals=bootstrap(gold,restored_pred,previous)
    fold_floors={'subjectivity_f1':.75,'aspect_f1':.75,'label_f1':.75,'polarity_f1':.83,
                 'objective_precision':.60,'objective_recall':.60}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='The least populated class')
        folds=list(StratifiedGroupKFold(4,shuffle=True,random_state=20261018).split(
            gold,gold.label+'/'+gold.aspect,groups=gold.review_id))
    fold_metrics=[]
    for n,(_,held) in enumerate(folds):
        fold_metrics.append({'fold':n,'metrics':metrics(gold.iloc[held],restored_pred.iloc[held])})
    unstable={k:{'worst_fold':min(f['metrics'][k] for f in fold_metrics),'floor':v}
              for k,v in fold_floors.items() if min(f['metrics'][k] for f in fold_metrics)<v}
    assert sha(freeze_path)==freeze_hash and sha(gold_path)==gold_hash
    assert sha(previnfo['model'])==previnfo['model_sha256']
    for name,digest in frozen['candidates'].items():assert sha(out/name/'model.joblib')==digest
    failed=chosen['failed_goals'];regress={k:m[k]-oldm[k] for k in GOALS if m[k]<oldm[k]-.01}
    report={'status':'completed_post_freeze_audit','selected_without_gold':selected,'gold_n':len(gold),
            'gold_sha256':gold_hash,'freeze_sha256':freeze_hash,'goals':GOALS,'selected_metrics':m,
            'previous_retained_metrics':oldm,'selected_delta_vs_previous':{k:m[k]-oldm[k] for k in GOALS},
            'selected_failed_goals':failed,'regressions_over_one_point':regress,'round_results':results,
            'fixed_prediction_grouped_folds':fold_metrics,'fold_floors':fold_floors,'fold_floor_failures':unstable,
            'selected_parent_bootstrap_2000':intervals,'per_class':checks,
            'verification':{'all_1073_reloaded_selected_predictions_match':True,'metrics_match_sklearn':True,
                            'gold_unchanged':True,'all_frozen_weights_unchanged':True,'frozen_selection_unchanged':True,
                            'previous_retained_model_unchanged':True,'weak_fitting_input_hashes_match':True},
            'release_status':'not_promoted_pending_independent_validation',
            'limitations':['Gold was excluded from fitting and selection in this run; earlier experiments already used this gold for development.',
                           'Weak-label and synthetic scores are proxies and can reward the original labeling rules.',
                           'The zero-shot model is an external pretrained model; no project gold was provided to it for training.',
                           'Conditional bootstrap intervals exclude historical selection, label uncertainty and distribution shift.',
                           'No candidate or threshold was changed in response to this audit.']}
    write_json(out/'gold_audit.json',report)
    errors=gold.add_prefix('gold_').join(restored_pred.add_prefix('pred_'))
    errors.loc[gold.aspect.ne(restored_pred.aspect)|gold.label.ne(restored_pred.label)].to_json(out/'selected_errors.jsonl',orient='records',lines=True,force_ascii=False)
    write_json(out/'release_gate.json',{'production_ready':False,'independent_test_passed':False,
                                      'point_quality_goals_passed':not failed,'grouped_fold_floors_passed':not unstable,
                                      'all_development_quality_gates_passed':not failed and not unstable,'regressions':regress,
                                      'previous_candidate_retained':True,'selected_without_gold':selected})
    labels={'subjectivity_f1':'主客观 macro-F1','aspect_f1':'方面 macro-F1','label_f1':'三分类 macro-F1',
            'polarity_f1':'极性 macro-F1','objective_precision':'客观句精确率','objective_recall':'客观句召回率',
            'minimum_aspect_f1':'最弱方面 F1','joint_accuracy':'方面和三分类同时正确率'}
    lines=['# 不用 gold 训练和选参的多方法实验','',f'状态：完成。预先锁定候选：`{selected}`。',
           '本轮训练、早停、阈值和模型选择均未读取项目 gold 答案。全部候选及选择结果先冻结，随后才运行一次 gold 对照审计。',
           '这份 gold 在历史实验中已参与开发，因此仍不是新的独立测试集。','',
           '## 锁定候选的实际表现','',
           '下表统一在全部 1,073 条现有 gold 上统计，不与此前 268 条监控分数混用。','',
           '| 指标 | 目标 | 上轮保留候选 | 本轮预先锁定候选 |','|---|---:|---:|---:|']
    for k,v in GOALS.items():lines.append(f'| {labels[k]} | {v:.1%} | {oldm[k]:.1%} | {m[k]:.1%} |')
    lines.extend(['','## 全部冻结候选的事后审计','',
                  '下面的成绩不反向用于修改模型、阈值或本轮选中指针。','',
                  '| 候选 | 方面 F1 | 三分类 F1 | 客观 precision | 客观 recall | 达到全部点估计目标 |',
                  '|---|---:|---:|---:|---:|---|'])
    for r in results:
        v=r['metrics'];lines.append(f"| {r['name']} | {v['aspect_f1']:.1%} | {v['label_f1']:.1%} | {v['objective_precision']:.1%} | {v['objective_recall']:.1%} | {r['all_point_targets_met']} |")
    lines.extend(['','## 验证与限制','',
                  '- 训练进程有文件读取拦截，拒绝访问 gold 答案和历史 gold 调参产物；唯一拒绝记录来自启动时的自检。',
                  '- weak train / weak dev 按父评论隔离，并排除原始 1,100 条评测句的父评论和规范化重复文本。',
                  '- 合成对照句只用于选参和诊断，未进入权重拟合；分数不能代表真实数据质量。',
                  '- 冻结后的选中权重重新加载，全部 1,073 条预测一致；核心 F1 与 sklearn 独立重算一致。',
                  '- 对固定预测另查四个父评论分组的表现，沿用每折 F1 与客观 precision/recall 下限，结果见 gold_audit.json。',
                  '- 仍保留上一轮候选；未自动构建或替换全量标签索引。','',
                  '## 来源与复跑','',
                  f'- 零样本模型：[模型卡](https://huggingface.co/{NLI_ID})。版本固定在训练 manifest。',
                  '- 新代码位于 `experiments/no_gold_optimization/`，运行方法见该目录 README。',
                  '- `monitor.json`：训练、选择、心跳及停止状态；`audit_monitor.json`：事后对照评估状态。',
                  '- `frozen_candidates.json`：读取 gold 之前锁定的权重哈希、脚本哈希和选中指针。',
                  '- `gold_audit.json`：完整指标、配对置信区间和逐类报告。',''])
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    status.update(status='completed',finished_at=datetime.now(timezone.utc).isoformat());phase('finished')
    print(json.dumps({'selected':selected,'metrics':m,'failed_goals':failed},ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,default=ROOT/'outputs/c_no_gold_optimization_20261018')
    main(p.parse_args().run_dir.resolve())
