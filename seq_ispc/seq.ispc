task void select_ispc_task( uniform int rowsPerTask,
							uniform int input[],
							uniform int output[])
{
	uniform int rowStart = taskIndex * rowsPerTask;
	uniform int rowStop = rowStart +  rowsPerTask;
    uniform int divider = 2;
	foreach(i = rowStart ... rowStop) {
		if (input[i] > 500 ) {
            output[i] = input[i];
        }
	}
}

export void select_ispc_withtasks( uniform int N,
								 uniform int input[],
							     uniform int output[])
{
	uniform int rowsPerTask = N / 4;
	launch[4] select_ispc_task(rowsPerTask, input, output);
}
