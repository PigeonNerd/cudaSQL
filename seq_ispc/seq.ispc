
task void select_ispc_task( uniform int rowsPerTask,
							uniform int input[],
							uniform int output[])
{
	uniform int rowStart = taskIndex * rowsPerTask;
	uniform int rowStop = rowStart +  rowsPerTask;
	foreach(i = rowStart ... rowStop) {
		output[i] = input[i] % 2 == 0 ? input[i] : 0;
	}
}

export void mandelbrot_ispc_withtasks( uniform int N,
								 uniform int input[],
							     uniform int output[])
{
	uniform int rowsPerTask = N / 512;
	launch[512] select_ispc_task(rowsPerTask, input, output);
}