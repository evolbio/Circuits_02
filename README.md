# Circuits_02: Overview

Julia source code for manuscript:

Frank, S. A. 2024. Circuit design in biology and machine learning. I. Anomaly detection. [arXiv:2411.15647](https://doi.org/10.48550/arXiv.2411.15647).

by Steven A. Frank, <https://stevefrank.org>.

The code and this file are on [GitHub](https://github.com/evolbio/Circuits_02.git).

This software provides the code to reproduce the figures in the manuscript.

# Getting started with the code

I outline the basic steps for installing Julia and running the code.

## Setup Julia

1. Install [Julia](https://julialang.org/) via the [juliaup](https://julialang.org/downloads/) program.
2. Download the source code for this project from this [GitHub](https://github.com/evolbio/Circuits_02.git) page. On that page, click on the **Code** button. The simplest option is to download and uncompress a zip archive. Alternatively, you will see options to clone the git repository if you want to do that.
3. I have tested the code only on MacOS but it should also run on Linux and Windows.

## Instantiate the environment for a project

1. In a terminal, go to the top directory of the code hierachy. In that directory is this file, README.md, LICENSE, and three subdirectories, Reservoir, Encoder, and CellTrend. Each subdirectory contains a [Julia project](https://docs.julialang.org/en/v1/manual/code-loading/#Project-environments). Each project produces one or more figures in the manuscript.

2. To run a project, change the current directory of your terminal to that directory. In this case, after downloading the code, change into the top directory of the downloaded code hierarchy.

3. In git, check the tags by typing `git tag --list`. Then checkout the appropriate tagged version. For example, if you want to match to a manuscript version, look at the version that is cited in the manuscript and use that version. Otherwise, use a git tool to pick version based on the date, or log, or just use the most recent version.

4. To start Julia for that project, type in the terminal `julia --project=.`

5. Once you have started Julia by the prior instruction, you are in the [REPL interactive command line](https://docs.julialang.org/en/v1/stdlib/REPL/) that interacts with the Julia software.

6. Each project depends on a set of external Julia packages that must be installed. To install or work with packages, you need to switch the REPL to package mode. At the `julia>` prompt, before typing any other character, hit the `]` key. You should now see a prompt `(Project) pkg>`. If you started in this project, it will show `(Anomaly) pkg>`. 

7. Type `instantiate`, which will install all the required packages listed in the file Project.toml and all the related dependencies. That file lists the specific package versions that I used and tested, creating the same environment. If you change package versions by updating or otherwise, then you may encounter errors that prevent the code from running. You can also see which version of the Julia software I used by looking at the top of the Manifest.toml file in the project directory. Usually, any Julia version greater than or equal to that version will work. However, if you have a problem, you may want to try matching to the Julia version in the Manifest file. You can do that by using [juliaup](https://julialang.org/downloads/) to download and activate a specific version of Julia.

8. Note that using a different version of Julia from the version listed in the Manifest may alter the sequence of random numbers even if seeded with the same values as in the example code. Also, running on different computer CPUs or different operating systems can alter results. Thus, it may not be possible to recreate the figures in the manuscript exactly.

9. Once instantiate has finished, which may take several minutes, you should return to the main julia prompt. To do that, immediately after a new `(Project) pkg>` prompt, hit the Backspace key (or CTRL-C might work). You should then once again see the main`julia>` prompt. If you get stuck at any time, you can exit Julia or open a new terminal and start Julia again as above.

10. You should only instantiate once for each project. The next time you start the same project, it should be ready immediately without moving to the package prompt. If you move to another project for the first time, you do need to repeat this process. Each project has its own environment and has to be instantiated the first time the project is used.

11. Note that if you update the Julia packages, then you may encounter problems or change the results for a given run with the same random seed. Updated packages do not guarantee to give the same results as earlier packages.

# Running the code in a project

In each project directory, there is a src subdirectory. In that subdirectory is a file, Run.jl. The code to reproduce one or more figures in the manuscript is in that Run.jl file.

Let's look at the current Anomaly project as an example. To run the code, first open a terminal, change directory to the the Reservoir directory, and then start Julia with `julia --project=.`

Then all that is needed to produce Figure 2 of the manuscript is to copy and paste the first lines from Run.jl, in particular copy and paste these lines (noting the warnings above about causes of differences):

```julia
using Anomaly, Plots, Random

rstate = Xoshiro(0xa242084f5b199be2, 0x53a1df5ce28618fd, 0x89250c8d5127cb4e, 0x6b5f53c46218257b, 0xc5ed5ca8a7cca17e);
sol,pl = temporal(;rstate=rstate);

# if the figure does not show on your screen, try:
display(pl);

# to savey the figure, replace the path with an appropriate one for your computer, using:
savefig(pl,"/Users/steve/Desktop/anomaly2.pdf")
```

This should produce Figure 2. The same procedure for the other project directories should produce the other figures in the manuscript.

# Understanding the code

There are not many comments in the code at present. To understand the code, look for the function call in Run.jl that calls a driver routine in the other source files. In the Anomaly project, the Run.jl code calls the driver() function in Anomaly. You can look at that function and follow the steps of the code. To understand what it is doing, you can look at the documentation for the any packages that are loaded and called. Check that you are looking at the right version of the documentation. On the documentation web page, in the lower left is the package version number. Match that to the package version number in the Manifest.toml file in the code on your computer.

It may take a bit of effort and reading to figure out how things work. But once you understand the basic concepts, you can use the code provided here as the basis for experimenting with other assumptions and applications.

# Notes

If you are going to modify the code, have a look at [Revise.jl](https://timholy.github.io/Revise.jl/stable/).

Occasionally there are specific directory location strings set within this source code that may not work on your computer. If you run into a problem, change the directory in the source code.
