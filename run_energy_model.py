"""Datasheet-based analytical energy budget for the sensor configurations.

NOT a measurement and NOT a prototype claim: average supply current is
estimated from vendor datasheet typicals with linear ODR scaling, so the
"redirect budget from inertial stacking / converter bits / clock rate toward
optical quality" recommendation becomes a quantified statement.

Constants (typical values, vendor datasheets):
  BMI270 accelerometer, normal mode: 0.45 mA at 100 Hz ODR  [27]
  BMI270 gyroscope,  normal mode: 0.55 mA at 100 Hz ODR  [27]
  BMM150 magnetometer, normal mode (10 Hz ODR): 0.50 mA, host-rate independent [28]
  MAX86176 PPG AFE: 0.30 mA quiescent + 0.90 mA at 256 Hz / 16-bit
            (LED duty + ADC), scaling with rate and resolution [26]
  nRF52840 SoC, periodic wake + DSP: 1.0 mA average [29]
  BLE radio: 5.0 mA for 1 ms per 20-byte payload connection event [29]
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

I_ACC_100 = 0.45
I_GYR_100 = 0.55
I_MAG = 0.50
AFE_Q = 0.30
AFE_FULL = 0.90
I_MCU = 1.0
I_BLE_MA = 5.0
BLE_EVENT_S = 0.001
BLE_PAYLOAD_BITS = 20 * 8


def imu_current(n_acc, n_gyro, n_mag, fs):
    return (n_acc * I_ACC_100 * fs / 100.0
            + n_gyro * I_GYR_100 * fs / 100.0
            + n_mag * I_MAG)


def afe_current(fs, bits):
    return AFE_Q + AFE_FULL * (fs / 256.0) * (0.5 + 0.5 * bits / 16.0)


def ble_current(fs, n_channels, bits):
    packets = fs * n_channels * bits / BLE_PAYLOAD_BITS
    return I_BLE_MA * BLE_EVENT_S * packets


CONFIGS = [
    ('full suite, 256 Hz, 16-bit', 2, 1, 1, 256, 16),
    ('acc+gyro, 256 Hz, 16-bit', 1, 1, 0, 256, 16),
    ('acc only, 256 Hz, 16-bit', 1, 0, 0, 256, 16),
    ('full suite, 16 Hz, 8-bit', 2, 1, 1, 16, 8),
    ('acc+gyro, 16 Hz, 8-bit', 1, 1, 0, 16, 8),
    ('acc only, 16 Hz, 8-bit', 1, 0, 0, 16, 8),
]


def main():
    rows = []
    base = None
    for name, na, ng, nm, fs, bits in CONFIGS:
        n_ch = na + ng + nm + 1          # + optical channel
        i_imu = imu_current(na, ng, nm, fs)
        i_afe = afe_current(fs, bits)
        i_ble = ble_current(fs, n_ch, bits)
        total = i_imu + i_afe + i_ble + I_MCU
        if base is None:
            base = total
        rows.append({'config': name, 'fs_hz': fs, 'bits': bits,
                     'i_imu_ma': round(i_imu, 3), 'i_afe_ma': round(i_afe, 3),
                     'i_ble_ma': round(i_ble, 3), 'i_mcu_ma': I_MCU,
                     'i_total_ma': round(total, 3),
                     'saving_vs_full_pct': round(100 * (1 - total / base), 1)})
    out = {'model': 'analytical, datasheet typicals, linear ODR scaling; '
                    'estimate only, no prototype and no measured power',
           'constants_mA': {'acc_per_100Hz': I_ACC_100, 'gyro_per_100Hz': I_GYR_100,
                            'mag': I_MAG, 'afe_quiescent': AFE_Q,
                            'afe_256Hz_16bit': AFE_FULL, 'mcu': I_MCU,
                            'ble_per_ms': I_BLE_MA},
           'rows': rows}
    with open(os.path.join(HERE, 'energy_model.json'), 'w') as fh:
        json.dump(out, fh, indent=1)
    for r in rows:
        print('%-28s imu=%.2f afe=%.2f ble=%.2f total=%.2f mA  save=%+.1f%%'
              % (r['config'], r['i_imu_ma'], r['i_afe_ma'], r['i_ble_ma'],
                 r['i_total_ma'], r['saving_vs_full_pct']))


if __name__ == '__main__':
    main()
