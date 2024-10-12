
import argparse
import json


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--log', type=str)
    args = parse.parse_args()
    log = args.log
    assert log != None
    with open(log,'r') as f:
        for r in f.readlines():
            if r.startswith('{'):
                result = json.loads(r)
                epoch = result['epoch']
                test_tnr_002 = result['test_tnr_0.002']
                test_tnr_005 = result['test_tnr_0.005']
                if 'test_ema_tnr_0.002' in result:
                    test_ema_tnr_002 = result.get('test_ema_tnr_0.002',0)
                    test_ema_tnr_005 = result.get('test_ema_tnr_0.005', 0)
                    print(f"epoch {epoch}，TNR@0.2%: {max(test_tnr_002['TNR'],test_ema_tnr_002['TNR'])} th: {test_tnr_002['th'] if test_tnr_002['TNR']>test_ema_tnr_002['TNR'] else test_ema_tnr_002['th']}，"
                          f"TNR@0.5%: {max(test_tnr_005['TNR'],test_ema_tnr_005['TNR'])} th: {test_tnr_005['th']  if test_tnr_005['TNR']>test_ema_tnr_005['TNR'] else test_ema_tnr_005['th']}")
                else:
                    print(f"epoch {epoch}，TNR@0.2%: {test_tnr_002['TNR']} th: {test_tnr_002['th'] }，TNR@0.5%: {test_tnr_005['TNR']} th: {test_tnr_005['th']}")



