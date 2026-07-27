OPENQASM 3.0;
include "stdgates.inc";
gate ecr _gate_q_0, _gate_q_1 {
  s _gate_q_0;
  sx _gate_q_1;
  cx _gate_q_0, _gate_q_1;
  x _gate_q_0;
}
rz(pi/2) $17;
sx $17;
rz(-0.7819715075232558) $17;
sx $17;
rz(-pi/2) $17;
rz(-pi/2) $25;
sx $25;
rz(-1.964936154074202) $25;
ecr $25, $17;
rz(-pi/2) $17;
sx $17;
rz(-2.3596211460665364) $17;
rz(2.7474528263104876) $25;
sx $25;
rz(pi/2) $25;
