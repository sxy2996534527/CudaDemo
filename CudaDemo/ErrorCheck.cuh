#pragma once

//��ĳЩ�첽��������֮�����첽�����Ψһ�����ǣ��ڵ���֮��ͨ������ cudaDeviceSynchronize()�����첽����ִ���������������κ�ͬ�����ƣ�����顣
//
//����ʱΪÿ����ʼ��Ϊ cudaSuccess �������߳�ά��һ��������������ڷ�������ʱ�ô�����븲�ǣ������ǲ�����֤�������첽���󣩡� cudaPeekAtLastError() �᷵�ش˱����� cudaGetLastError() ���ش˱�������������Ϊ cudaSuccess��
//
//�ں������������κδ�����룬��˱������ں��������������� cudaPeekAtLastError() �� cudaGetLastError() ��ȷ���κ�����ǰ����
//Ϊ��ȷ�� cudaPeekAtLastError() �� cudaGetLastError() ���ص��κδ����������ں�����֮ǰ�ĵ��ã�����ȷ�����ں�����֮ǰ�� CUDA ����ʱ�Ĵ����������Ϊ cudaSuccess�����磬�������ں�����֮ǰ����cudaGetLastError() ���ں��������첽�ģ����Ҫ����첽����Ӧ�ó���������ں������� cudaPeekAtLastError() �� cudaGetLastError() �ĵ���֮�����ͬ����
//
//��ע�⣬cudaStreamQuery() �� cudaEventQuery() ���ܷ��� cudaErrorNotReady ����������Ϊ������� cudaPeekAtLastError() �� cudaGetLastError() ���ᱨ�档