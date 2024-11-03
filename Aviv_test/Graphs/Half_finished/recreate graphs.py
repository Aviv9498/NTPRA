import numpy as np
import matplotlib.pyplot as plt


def recreate_half_finished_N_15_E_25():

    # Adjusted step heights and extended horizontal lines to match the reference image

    # Step heights after the dashed line for each series
    corrected_step_data = {
        "Time-Series 10 Flows": {"x": [0, 35, 120], "y": [260, 260, 265]},  # Step up at 35 iterations
        "Single 10 Flows": {"x": [0, 35, 120], "y": [260, 260, 248]},  # Step down at 35 iterations
        "Time-Series 20 Flows": {"x": [0, 70, 120], "y": [140, 140, 173]},  # Step up at 70 iterations
        "Single 20 Flows": {"x": [0, 70, 120], "y": [140, 140, 125]},  # Step down at 70 iterations
        "Time-Series 30 Flows": {"x": [0, 110, 120], "y": [90, 90, 120]},  # Step up at 110 iterations
        "Single 30 Flows": {"x": [0, 110, 120], "y": [90, 90, 85]},  # Step down at 110 iterations
    }

    # Plotting the corrected steps and extended horizontal lines
    plt.figure()

    """
    10 Flows
    """

    # Time-Series 10 Flows
    plt.plot(corrected_step_data["Time-Series 10 Flows"]["x"][:2], corrected_step_data["Time-Series 10 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([35, 35], [260, 265], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 10 Flows"]["x"][1:], [265, 265], '-', color='brown', label='Time-Series 10 Flows')  # Horizontal line

    # Single 10 Flows
    plt.plot(corrected_step_data["Single 10 Flows"]["x"][:2], corrected_step_data["Single 10 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([35, 35], [260, 248], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 10 Flows"]["x"][1:], [248, 248], '-', color='goldenrod', label='Single 10 Flows')  # Horizontal line

    """
    20 Flows
    """

    # Time-Series 20 Flows
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][:2], corrected_step_data["Time-Series 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([70, 70], [140, 173], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][1:], [173, 173], '-', color='lime', label='Time-Series 20 Flows')  # Horizontal line

    # Single 20 Flows
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][:2], corrected_step_data["Single 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([70, 70], [140, 125], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][1:], [125, 125], '-', color='orange', label='Single 20 Flows')  # Horizontal line

    """
    30 Flows
    """

    # Time-Series 30 Flows
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][:2], corrected_step_data["Time-Series 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([110, 110], [90, 120], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][1:], [120, 120], '-', color='purple', label='Time-Series 30 Flows')  # Horizontal line

    # Single 30 Flows
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][:2], corrected_step_data["Single 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([110, 110], [90, 85], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][1:], [85, 85], '-', color='teal', label='Single 30 Flows')  # Horizontal line

    # Labels and title
    plt.xlabel('Average Iteration Number')
    plt.ylabel('Average Rate')
    plt.title('Average Rate With Half Flows \'Alive\'')

    # Legend
    plt.legend(loc='lower left')

    plt.grid(True)

    # Save the plot as an image
    plt.savefig('average_rate_plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()

def recreate_half_finished_N_15_E_20():

    # Adjusted step heights and extended horizontal lines to match the reference image

    # Step heights after the dashed line for each series
    corrected_step_data = {
        "Time-Series 10 Flows": {"x": [0, 80, 200], "y": [125, 125, 110]},  # Step up at 35 iterations
        "Single 10 Flows": {"x": [0, 80, 200], "y": [125, 125, 101]},  # Step down at 35 iterations

        "Time-Series 20 Flows": {"x": [0, 140, 200], "y": [80.5, 80.5, 73]},  # Step up at 70 iterations
        "Single 20 Flows": {"x": [0, 140, 200], "y": [80.5, 80.5, 60]},  # Step down at 70 iterations

        "Time-Series 30 Flows": {"x": [0, 190, 200], "y": [61, 61, 59]},  # Step up at 110 iterations
        "Single 30 Flows": {"x": [0, 190, 200], "y": [61, 61, 52]},  # Step down at 110 iterations
    }

    # Plotting the corrected steps and extended horizontal lines
    plt.figure()

    """
    10 Flows
    """

    # Time-Series 10 Flows
    plt.plot(corrected_step_data["Time-Series 10 Flows"]["x"][:2], corrected_step_data["Time-Series 10 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([80, 80], [125, 110], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 10 Flows"]["x"][1:], [110, 110], '-', color='brown', label='Time-Series 10 Flows')  # Horizontal line

    # Single 10 Flows
    plt.plot(corrected_step_data["Single 10 Flows"]["x"][:2], corrected_step_data["Single 10 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([80, 80], [125, 101], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 10 Flows"]["x"][1:], [101, 101], '-', color='goldenrod', label='Single 10 Flows')  # Horizontal line

    """
    20 Flows
    """

    # Time-Series 20 Flows
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][:2], corrected_step_data["Time-Series 20 Flows"]["y"][:2],
             '--', color='blue')  # Dashed line
    plt.plot([140, 140], [80.5, 73], '--', color="blue")  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][1:], [73, 73], '-', color='lime', label='Time-Series 20 Flows')  # Horizontal line

    # Single 20 Flows
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][:2], corrected_step_data["Single 20 Flows"]["y"][:2],
             '--', color='blue')  # Dashed line
    plt.plot([140, 140], [80.5, 60], '--', color='blue')  # Vertical step down
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][1:], [60, 60], '-', color='orange', label='Single 20 Flows')  # Horizontal line

    """
    30 Flows
    """

    # Time-Series 30 Flows
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][:2], corrected_step_data["Time-Series 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([190, 190], [61, 59], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][1:], [59, 59], '-', color='purple', label='Time-Series 30 Flows')  # Horizontal line

    # Single 30 Flows
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][:2], corrected_step_data["Single 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([190, 190], [61, 52], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][1:], [52, 52], '-', color='teal', label='Single 30 Flows')  # Horizontal line

    # Labels and title
    plt.xlabel('Average Iteration Number')
    plt.ylabel('Average Rate')
    plt.title('Average Rate With Half Flows \'Alive\'')

    # Legend
    plt.legend(loc='lower left')

    plt.grid(True)

    # Save the plot as an image
    plt.savefig('average_rate_plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def recreate_half_finished_N_5_E_10():

    # Adjusted step heights and extended horizontal lines to match the reference image

    # Step heights after the dashed line for each series
    corrected_step_data = {
        "Time-Series 5 Flows": {"x": [0, 38, 70], "y": [155, 155, 195]},  # Step up at 35 iterations
        "Single 5 Flows": {"x": [0, 38, 70], "y": [155, 155, 163]},  # Step down at 35 iterations

        "Time-Series 10 Flows": {"x": [0, 63, 70], "y": [130, 130, 149]},  # Step up at 70 iterations
        "Single 10 Flows": {"x": [0, 63, 70], "y": [130, 130, 118]},  # Step down at 70 iterations

        "Time-Series 20 Flows": {"x": [0, 55, 70], "y": [92, 92, 98]},  # Step up at 110 iterations
        "Single 20 Flows": {"x": [0, 55, 70], "y": [92, 92, 83]},  # Step down at 110 iterations
    }

    # Plotting the corrected steps and extended horizontal lines
    plt.figure()

    """
    5 Flows
    """

    # Time-Series 5 Flows
    plt.plot(corrected_step_data["Time-Series 5 Flows"]["x"][:2], corrected_step_data["Time-Series 5 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([38, 38], [155, 195], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 5 Flows"]["x"][1:], [195, 195], '-', color='brown', label='Time-Series 5 Flows')  # Horizontal line

    # Single 5 Flows
    plt.plot(corrected_step_data["Single 5 Flows"]["x"][:2], corrected_step_data["Single 5 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([38, 38], [155, 163], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 5 Flows"]["x"][1:], [163, 163], '-', color='goldenrod', label='Single 5 Flows')  # Horizontal line

    """
    10 Flows
    """

    # Time-Series 10 Flows
    plt.plot(corrected_step_data["Time-Series 10 Flows"]["x"][:2], corrected_step_data["Time-Series 10 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([63, 63], [130, 149], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 10 Flows"]["x"][1:], [149, 149], '-', color='lime', label='Time-Series 10 Flows')  # Horizontal line

    # Single 10 Flows
    plt.plot(corrected_step_data["Single 10 Flows"]["x"][:2], corrected_step_data["Single 10 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([63, 63], [130, 118], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 10 Flows"]["x"][1:], [118, 118], '-', color='orange', label='Single 10 Flows')  # Horizontal line

    """
    20 Flows
    """

    # Time-Series 20 Flows
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][:2], corrected_step_data["Time-Series 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([55, 55], [92, 98], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][1:], [98, 98], '-', color='purple', label='Time-Series 20 Flows')  # Horizontal line

    # Single 20 Flows
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][:2], corrected_step_data["Single 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([55, 55], [92, 83], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][1:], [83, 83], '-', color='teal', label='Single 20 Flows')  # Horizontal line

    # Labels and title
    plt.xlabel('Average Iteration Number')
    plt.ylabel('Average Rate')
    plt.title('Average Rate With Half Flows \'Alive\'')

    # Legend
    plt.legend(loc='lower left')

    plt.grid(True)

    # Save the plot as an image
    plt.savefig('average_rate_plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def recreate_half_finished_N_60_E_70():

    # Adjusted step heights and extended horizontal lines to match the reference image

    # Step heights after the dashed line for each series
    corrected_step_data = {
        "Time-Series 20 Flows": {"x": [0, 25, 85], "y": [67, 67, 65]},  # Step up at 35 iterations
        "Single 20 Flows": {"x": [0, 25, 85], "y": [67, 67, 45]},  # Step down at 35 iterations

        "Time-Series 30 Flows": {"x": [0, 39, 85], "y": [45, 45, 41]},  # Step up at 70 iterations
        "Single 30 Flows": {"x": [0, 39, 85], "y": [45, 45, 31]},  # Step down at 70 iterations

        "Time-Series 40 Flows": {"x": [0, 57, 85], "y": [32, 32, 29]},  # Step up at 110 iterations
        "Single 40 Flows": {"x": [0, 57, 85], "y": [32, 32, 22]},  # Step down at 110 iterations

        "Time-Series 50 Flows": {"x": [0, 79, 85], "y": [25, 25, 24]},  # Step up at 110 iterations
        "Single 50 Flows": {"x": [0, 79, 85], "y": [25, 25, 18]},  # Step down at 110 iterations
    }

    # Plotting the corrected steps and extended horizontal lines
    plt.figure()

    """
    20 Flows
    """

    # Time-Series 5 Flows
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][:2], corrected_step_data["Time-Series 20 Flows"]["y"][:2],
             '--', color='blue')  # Dashed line
    plt.plot([25, 25], [67, 65], '--', color='blue')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][1:], [65, 65], '-', color='brown', label='Time-Series 20 Flows')  # Horizontal line

    # Single 5 Flows
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][:2], corrected_step_data["Single 20 Flows"]["y"][:2],
             '--', color='blue')  # Dashed line
    plt.plot([25, 25], [67, 45], '--', color='blue')  # Vertical step down
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][1:], [45, 45], '-', color='goldenrod', label='Single 20 Flows')  # Horizontal line

    """
    30 Flows
    """

    # Time-Series 10 Flows
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][:2], corrected_step_data["Time-Series 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([39, 39], [45, 41], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][1:], [41, 41], '-', color='lime', label='Time-Series 30 Flows')  # Horizontal line

    # Single 10 Flows
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][:2], corrected_step_data["Single 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([39, 39], [45, 31], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][1:], [31, 31], '-', color='orange', label='Single 30 Flows')  # Horizontal line

    """
    40 Flows
    """

    # Time-Series 20 Flows
    plt.plot(corrected_step_data["Time-Series 40 Flows"]["x"][:2], corrected_step_data["Time-Series 40 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([57, 57], [32, 29], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 40 Flows"]["x"][1:], [29, 29], '-', color='purple', label='Time-Series 40 Flows')  # Horizontal line

    # Single 20 Flows
    plt.plot(corrected_step_data["Single 40 Flows"]["x"][:2], corrected_step_data["Single 40 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([57, 57], [32, 22], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 40 Flows"]["x"][1:], [22, 22], '-', color='teal', label='Single 40 Flows')  # Horizontal line

    """
    50 Flows
    """

    # Time-Series 20 Flows
    plt.plot(corrected_step_data["Time-Series 50 Flows"]["x"][:2], corrected_step_data["Time-Series 50 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([79, 79], [25, 24], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 50 Flows"]["x"][1:], [24, 24], '-', color='red', label='Time-Series 50 Flows')  # Horizontal line

    # Single 20 Flows
    plt.plot(corrected_step_data["Single 50 Flows"]["x"][:2], corrected_step_data["Single 50 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([79, 79], [25, 18], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 50 Flows"]["x"][1:], [18, 18], '-', color='darkblue', label='Single 50 Flows')  # Horizontal line

    # Labels and title
    plt.xlabel('Average Iteration Number')
    plt.ylabel('Average Rate')
    plt.title('Average Rate With Half Flows \'Alive\'')

    # Legend
    plt.legend(loc='lower left')

    plt.grid(True)

    # Save the plot as an image
    plt.savefig('average_rate_plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def recreate_half_finished_N_40_E_50():

    # Adjusted step heights and extended horizontal lines to match the reference image

    # Step heights after the dashed line for each series
    corrected_step_data = {
        "Time-Series 20 Flows": {"x": [0, 20, 35], "y": [81, 81, 78]},  # Step up at 35 iterations
        "Single 20 Flows": {"x": [0, 20, 35], "y": [81, 81, 60]},  # Step down at 35 iterations

        "Time-Series 30 Flows": {"x": [0, 33, 35], "y": [54, 54, 53]},  # Step up at 70 iterations
        "Single 30 Flows": {"x": [0, 33, 35], "y": [54, 54, 41]},  # Step down at 70 iterations
    }

    # Plotting the corrected steps and extended horizontal lines
    plt.figure()

    """
    20 Flows
    """

    # Time-Series 20 Flows
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][:2], corrected_step_data["Time-Series 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([20, 20], [81, 78], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][1:], [78, 78], '-', color='brown', label='Time-Series 20 Flows')  # Horizontal line

    # Single 20 Flows
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][:2], corrected_step_data["Single 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([20, 20], [81, 60], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][1:], [60, 60], '-', color='goldenrod', label='Single 20 Flows')  # Horizontal line

    """
    30 Flows
    """

    # Time-Series 30 Flows
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][:2], corrected_step_data["Time-Series 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([33, 33], [54, 53], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][1:], [53, 53], '-', color='lime', label='Time-Series 30 Flows')  # Horizontal line

    # Single 30 Flows
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][:2], corrected_step_data["Single 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([33, 33], [54, 41], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][1:], [41, 41], '-', color='orange', label='Single 30 Flows')  # Horizontal line


    # Labels and title
    plt.xlabel('Average Iteration Number')
    plt.ylabel('Average Rate')
    plt.title('Average Rate With Half Flows \'Alive\'')

    # Legend
    plt.legend(loc='lower left')

    plt.grid(True)

    # Save the plot as an image
    plt.savefig('average_rate_plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


def recreate_half_finished_N_35_E_45():

    # Adjusted step heights and extended horizontal lines to match the reference image

    # Step heights after the dashed line for each series
    corrected_step_data = {
        "Time-Series 20 Flows": {"x": [0, 18, 35], "y": [84, 84, 79]},  # Step up at 35 iterations
        "Single 20 Flows": {"x": [0, 18, 35], "y": [84, 84, 62]},  # Step down at 35 iterations

        "Time-Series 30 Flows": {"x": [0, 29, 35], "y": [56, 56, 55.5]},  # Step up at 70 iterations
        "Single 30 Flows": {"x": [0, 29, 35], "y": [56, 56, 45.5]},  # Step down at 70 iterations
    }

    # Plotting the corrected steps and extended horizontal lines
    plt.figure()

    """
    20 Flows
    """

    # Time-Series 20 Flows
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][:2], corrected_step_data["Time-Series 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([18, 18], [84, 79], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 20 Flows"]["x"][1:], [79, 79], '-', color='brown', label='Time-Series 20 Flows')  # Horizontal line

    # Single 20 Flows
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][:2], corrected_step_data["Single 20 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([18, 18], [84, 62], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 20 Flows"]["x"][1:], [62, 62], '-', color='goldenrod', label='Single 20 Flows')  # Horizontal line

    """
    30 Flows
    """

    # Time-Series 30 Flows
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][:2], corrected_step_data["Time-Series 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([29, 29], [56, 55.5], '--', color='black')  # Vertical step up
    plt.plot(corrected_step_data["Time-Series 30 Flows"]["x"][1:], [55.5, 55.5], '-', color='lime', label='Time-Series 30 Flows')  # Horizontal line

    # Single 30 Flows
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][:2], corrected_step_data["Single 30 Flows"]["y"][:2],
             '--', color='black')  # Dashed line
    plt.plot([29, 29], [56, 45.5], '--', color='black')  # Vertical step down
    plt.plot(corrected_step_data["Single 30 Flows"]["x"][1:], [45.5, 45.5], '-', color='orange', label='Single 30 Flows')  # Horizontal line


    # Labels and title
    plt.xlabel('Average Iteration Number')
    plt.ylabel('Average Rate')
    plt.title('Average Rate With Half Flows \'Alive\'')

    # Legend
    plt.legend(loc='lower left')

    plt.grid(True)

    # Save the plot as an image
    plt.savefig('average_rate_plot.png', dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    # recreate_half_finished_N_15_E_25()
    # recreate_half_finished_N_15_E_20()
    # recreate_half_finished_N_5_E_10()
    # recreate_half_finished_N_60_E_70()
    # recreate_half_finished_N_40_E_50()
    recreate_half_finished_N_35_E_45()